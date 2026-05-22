"""
UC2 — Vietnamese Sentiment Analysis — Apache Airflow
Model: vinai/phobert-base + classification head Linear(768→3)
Dataset: UIT-VSFC, stratified 500 samples (train), full test set
Pipeline: load_data >> preprocess >> train_model >> evaluate_model
"""

import sys
sys.path.insert(0, '/home/airflow/airflowServer/dags/shared')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}


# ── Step 1: Load data ────────────────────────────────────────────────────────
def load_data(**kwargs):
    import sys
    sys.path.insert(0, '/home/airflow/airflowServer/dags/shared')

    import time
    import pickle
    import os
    from datasets import load_dataset
    from config_phobert import DATASET_NAME, SAMPLE_SIZE, SEED, LABEL_COL
    from sampling_utils import stratified_sample

    t0 = time.time()
    print("[load_data] Loading UIT-VSFC from HuggingFace...")

    dataset  = load_dataset(DATASET_NAME)
    train_ds = dataset['train']
    val_ds   = dataset['validation']
    test_ds  = dataset['test']

    print(f"[load_data] Full — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Stratified sampling 500 mẫu từ train — val và test giữ nguyên
    train_sampled = stratified_sample(train_ds, LABEL_COL, SAMPLE_SIZE, seed=SEED)
    print(f"[load_data] Sampled train: {len(train_sampled)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    os.makedirs('/tmp/vsfc', exist_ok=True)
    train_path = '/tmp/vsfc/train.pkl'
    val_path   = '/tmp/vsfc/val.pkl'
    test_path  = '/tmp/vsfc/test.pkl'

    with open(train_path, 'wb') as f: pickle.dump(train_sampled, f)
    with open(val_path,   'wb') as f: pickle.dump(val_ds,        f)
    with open(test_path,  'wb') as f: pickle.dump(test_ds,       f)

    elapsed = round(time.time() - t0, 3)
    print(f"[load_data] done. time={elapsed}s")

    ti = kwargs['ti']
    ti.xcom_push(key='train_path', value=train_path)
    ti.xcom_push(key='val_path',   value=val_path)
    ti.xcom_push(key='test_path',  value=test_path)
    ti.xcom_push(key='load_time',  value=elapsed)


# ── Step 2: Preprocess ───────────────────────────────────────────────────────
def preprocess(**kwargs):
    import sys
    sys.path.insert(0, '/home/airflow/airflowServer/dags/shared')

    import time
    import pickle
    from transformers import AutoTokenizer
    from config_phobert import PHOBERT_MODEL_NAME, MAX_LENGTH, TEXT_COL, LABEL_COL

    ti         = kwargs['ti']
    train_path = ti.xcom_pull(key='train_path', task_ids='load_data')
    val_path   = ti.xcom_pull(key='val_path',   task_ids='load_data')
    test_path  = ti.xcom_pull(key='test_path',  task_ids='load_data')

    t0 = time.time()
    print("[preprocess] Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    def tokenize(dataset):
        tokenized = dataset.map(
            lambda x: tokenizer(
                x[TEXT_COL],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
        )
        tokenized = tokenized.remove_columns(
            [c for c in tokenized.column_names
             if c not in ['input_ids', 'attention_mask', LABEL_COL]]
        )
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', LABEL_COL])
        return tokenized

    with open(train_path, 'rb') as f: train_ds = pickle.load(f)
    with open(val_path,   'rb') as f: val_ds   = pickle.load(f)
    with open(test_path,  'rb') as f: test_ds  = pickle.load(f)

    print("[preprocess] Tokenizing...")
    train_tok = tokenize(train_ds)
    val_tok   = tokenize(val_ds)
    test_tok  = tokenize(test_ds)

    tok_train_path = '/tmp/vsfc/train_tok.pkl'
    tok_val_path   = '/tmp/vsfc/val_tok.pkl'
    tok_test_path  = '/tmp/vsfc/test_tok.pkl'

    with open(tok_train_path, 'wb') as f: pickle.dump(train_tok, f)
    with open(tok_val_path,   'wb') as f: pickle.dump(val_tok,   f)
    with open(tok_test_path,  'wb') as f: pickle.dump(test_tok,  f)

    elapsed = round(time.time() - t0, 3)
    print(f"[preprocess] done. time={elapsed}s")

    ti.xcom_push(key='tok_train_path',  value=tok_train_path)
    ti.xcom_push(key='tok_val_path',    value=tok_val_path)
    ti.xcom_push(key='tok_test_path',   value=tok_test_path)
    ti.xcom_push(key='preprocess_time', value=elapsed)


# ── Step 3: Fine-tune PhoBERT ────────────────────────────────────────────────
def train_model(**kwargs):
    import sys
    sys.path.insert(0, '/home/airflow/airflowServer/dags/shared')

    import time
    import pickle
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from torch.optim import AdamW
    from config_phobert import PHOBERT_MODEL_NAME, NUM_LABELS, HPARAMS, LABEL_COL, SEED

    ti             = kwargs['ti']
    tok_train_path = ti.xcom_pull(key='tok_train_path', task_ids='preprocess')
    tok_val_path   = ti.xcom_pull(key='tok_val_path',   task_ids='preprocess')

    with open(tok_train_path, 'rb') as f: train_ds = pickle.load(f)
    with open(tok_val_path,   'rb') as f: val_ds   = pickle.load(f)

    LR           = HPARAMS['lr']
    BATCH_SIZE   = HPARAMS['batch_size']
    EPOCHS       = HPARAMS['epochs']
    WEIGHT_DECAY = HPARAMS['weight_decay']
    ACCUM_STEPS  = HPARAMS['gradient_accumulation_steps']

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)
            if HPARAMS.get('gradient_checkpointing'):
                self.phobert.gradient_checkpointing_enable()

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    torch.manual_seed(SEED)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] device={device}")

    model     = PhoBERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"[train] epochs={EPOCHS} lr={LR} batch={BATCH_SIZE} accum={ACCUM_STEPS}")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch[LABEL_COL].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch[LABEL_COL].to(device)
                val_loss      += criterion(model(input_ids, attention_mask), labels).item()

        print(f"  Epoch {epoch+1}/{EPOCHS} — train_loss: {avg_loss:.4f}, val_loss: {val_loss/len(val_loader):.4f}")

    train_time = round(time.time() - t0, 3)
    print(f"[train] done. train_time={train_time}s")

    model_path = '/tmp/vsfc/phobert_finetuned.pth'
    torch.save(model.state_dict(), model_path)

    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='train_time', value=train_time)


# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
def evaluate_model(**kwargs):
    import sys
    sys.path.insert(0, '/home/airflow/airflowServer/dags/shared')

    import time
    import pickle
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from sklearn.metrics import accuracy_score, f1_score
    from config_phobert import PHOBERT_MODEL_NAME, NUM_LABELS, HPARAMS, LABEL_COL

    ti              = kwargs['ti']
    tok_test_path   = ti.xcom_pull(key='tok_test_path',   task_ids='preprocess')
    model_path      = ti.xcom_pull(key='model_path',      task_ids='train_model')
    train_time      = ti.xcom_pull(key='train_time',      task_ids='train_model')
    load_time       = ti.xcom_pull(key='load_time',       task_ids='load_data')
    preprocess_time = ti.xcom_pull(key='preprocess_time', task_ids='preprocess')

    BATCH_SIZE = HPARAMS['batch_size']

    with open(tok_test_path, 'rb') as f: test_ds = pickle.load(f)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PhoBERTClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch[LABEL_COL]
            preds          = model(input_ids, attention_mask).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    eval_time = round(time.time() - t0, 3)
    accuracy  = round(accuracy_score(all_labels, all_preds) * 100, 4)
    f1        = round(f1_score(all_labels, all_preds, average='macro') * 100, 4)
    total     = round(load_time + preprocess_time + train_time + eval_time, 3)

    print(f"[evaluate] Results:")
    print(f"  Accuracy:        {accuracy:.4f}%")
    print(f"  F1-macro:        {f1:.4f}%")
    print(f"  Load time:       {load_time}s")
    print(f"  Preprocess time: {preprocess_time}s")
    print(f"  Train time:      {train_time}s")
    print(f"  Eval time:       {eval_time}s")
    print(f"  Total time:      {total}s")

    ti.xcom_push(key='accuracy',   value=accuracy)
    ti.xcom_push(key='f1',         value=f1)
    ti.xcom_push(key='eval_time',  value=eval_time)
    ti.xcom_push(key='total_time', value=total)


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id='vietnamese_sentiment_airflow',
    default_args=default_args,
    description='UC2 PhoBERT — UIT-VSFC 500 samples stratified, 3 epochs',
    schedule_interval=None,
    catchup=False,
    tags=['phobert', 'sentiment', 'vietnamese', 'uc2'],
) as dag:

    t1 = PythonOperator(task_id='load_data',      python_callable=load_data)
    t2 = PythonOperator(task_id='preprocess',     python_callable=preprocess)
    t3 = PythonOperator(task_id='train_model',    python_callable=train_model)
    t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)

    t1 >> t2 >> t3 >> t4