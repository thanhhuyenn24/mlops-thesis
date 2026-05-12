"""
UC2 — Vietnamese Sentiment Analysis — Apache Airflow
Model: vinai/phobert-base + classification head Linear(768→3)
Dataset: UIT-VSFC full (train/val/test split chuẩn)
Pipeline: load_data >> preprocess >> train_model >> evaluate_model
"""

import sys
sys.path.insert(0, '/opt/airflow/dags/shared')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

# ── Config từ shared file ─────────────────────────────────────────────────────
PHOBERT_MODEL_NAME = "vinai/phobert-base"
NUM_LABELS  = 3
MAX_LENGTH  = 256
LR          = 2e-5
BATCH_SIZE  = 16
EPOCHS      = 3
WEIGHT_DECAY = 0.01
DATASET_NAME = "uitnlp/vietnamese_students_feedback"
TEXT_COL    = "sentence"
LABEL_COL   = "sentiment"


# ── Step 1: Load data ────────────────────────────────────────────────────────
def load_data(**kwargs):
    import time
    from datasets import load_dataset
    import pickle, os

    t0 = time.time()
    print("[load_data] Loading UIT-VSFC from HuggingFace...")

    # Dùng đúng split chuẩn của dataset — không tự chia lại
    dataset = load_dataset(DATASET_NAME)
    train_ds = dataset['train']
    val_ds   = dataset['validation']
    test_ds  = dataset['test']

    print(f"[load_data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Lưu xuống disk để pass giữa tasks
    os.makedirs('/tmp/vsfc', exist_ok=True)
    train_path = '/tmp/vsfc/train.pkl'
    val_path   = '/tmp/vsfc/val.pkl'
    test_path  = '/tmp/vsfc/test.pkl'

    with open(train_path, 'wb') as f: pickle.dump(train_ds, f)
    with open(val_path,   'wb') as f: pickle.dump(val_ds,   f)
    with open(test_path,  'wb') as f: pickle.dump(test_ds,  f)

    elapsed = round(time.time() - t0, 3)
    print(f"[load_data] done. time={elapsed}s")

    ti = kwargs['ti']
    ti.xcom_push(key='train_path', value=train_path)
    ti.xcom_push(key='val_path',   value=val_path)
    ti.xcom_push(key='test_path',  value=test_path)
    ti.xcom_push(key='load_time',  value=elapsed)


# ── Step 2: Preprocess ───────────────────────────────────────────────────────
def preprocess(**kwargs):
    import time
    import pickle
    from transformers import AutoTokenizer

    ti         = kwargs['ti']
    train_path = ti.xcom_pull(key='train_path', task_ids='load_data')
    val_path   = ti.xcom_pull(key='val_path',   task_ids='load_data')
    test_path  = ti.xcom_pull(key='test_path',  task_ids='load_data')

    t0 = time.time()
    print("[preprocess] Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    def tokenize(dataset):
        return dataset.map(
            lambda x: tokenizer(
                x[TEXT_COL],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
        )

    with open(train_path, 'rb') as f: train_ds = pickle.load(f)
    with open(val_path,   'rb') as f: val_ds   = pickle.load(f)
    with open(test_path,  'rb') as f: test_ds  = pickle.load(f)

    print("[preprocess] Tokenizing...")
    train_tok = tokenize(train_ds)
    val_tok   = tokenize(val_ds)
    test_tok  = tokenize(test_ds)

    # Set format cho PyTorch
    cols = ['input_ids', 'attention_mask', LABEL_COL]
    train_tok.set_format(type='torch', columns=cols)
    val_tok.set_format(type='torch',   columns=cols)
    test_tok.set_format(type='torch',  columns=cols)

    # Lưu lại
    tok_train_path = '/tmp/vsfc/train_tok.pkl'
    tok_val_path   = '/tmp/vsfc/val_tok.pkl'
    tok_test_path  = '/tmp/vsfc/test_tok.pkl'

    with open(tok_train_path, 'wb') as f: pickle.dump(train_tok, f)
    with open(tok_val_path,   'wb') as f: pickle.dump(val_tok,   f)
    with open(tok_test_path,  'wb') as f: pickle.dump(test_tok,  f)

    elapsed = round(time.time() - t0, 3)
    print(f"[preprocess] done. time={elapsed}s")

    ti.xcom_push(key='tok_train_path', value=tok_train_path)
    ti.xcom_push(key='tok_val_path',   value=tok_val_path)
    ti.xcom_push(key='tok_test_path',  value=tok_test_path)
    ti.xcom_push(key='preprocess_time', value=elapsed)


# ── Step 3: Fine-tune PhoBERT ────────────────────────────────────────────────
def train_model(**kwargs):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from torch.optim import AdamW
    import pickle, time

    ti             = kwargs['ti']
    tok_train_path = ti.xcom_pull(key='tok_train_path', task_ids='preprocess')
    tok_val_path   = ti.xcom_pull(key='tok_val_path',   task_ids='preprocess')

    with open(tok_train_path, 'rb') as f: train_ds = pickle.load(f)
    with open(tok_val_path,   'rb') as f: val_ds   = pickle.load(f)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # PhoBERT + classification head — giống MLflow
    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]   # [CLS] token
            return self.classifier(cls)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] device={device}")

    model     = PhoBERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"[train] Fine-tuning PhoBERT — epochs={EPOCHS} lr={LR} batch={BATCH_SIZE}")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch[LABEL_COL].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch[LABEL_COL].to(device)
                logits         = model(input_ids, attention_mask)
                val_loss      += criterion(logits, labels).item()

        print(f"  Epoch {epoch+1}/{EPOCHS} — train_loss: {avg_loss:.4f}, val_loss: {val_loss/len(val_loader):.4f}")

    train_time = round(time.time() - t0, 3)
    print(f"[train] done. train_time={train_time}s")

    model_path = '/tmp/vsfc/phobert_finetuned.pth'
    torch.save(model.state_dict(), model_path)

    ti.xcom_push(key='model_path',  value=model_path)
    ti.xcom_push(key='train_time',  value=train_time)


# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
def evaluate_model(**kwargs):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from sklearn.metrics import accuracy_score, f1_score
    import pickle, time

    ti            = kwargs['ti']
    tok_test_path = ti.xcom_pull(key='tok_test_path', task_ids='preprocess')
    model_path    = ti.xcom_pull(key='model_path',    task_ids='train_model')
    train_time    = ti.xcom_pull(key='train_time',    task_ids='train_model')
    load_time     = ti.xcom_pull(key='load_time',     task_ids='load_data')
    preprocess_time = ti.xcom_pull(key='preprocess_time', task_ids='preprocess')

    with open(tok_test_path, 'rb') as f: test_ds = pickle.load(f)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert     = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier  = nn.Linear(768, NUM_LABELS)

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls)

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
            logits         = model(input_ids, attention_mask)
            preds          = torch.argmax(logits, dim=1).cpu()
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

    ti.xcom_push(key='accuracy',  value=accuracy)
    ti.xcom_push(key='f1',        value=f1)
    ti.xcom_push(key='eval_time', value=eval_time)
    ti.xcom_push(key='total_time', value=total)


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id='vietnamese_sentiment_airflow',
    default_args=default_args,
    description='UC2 PhoBERT fine-tune — UIT-VSFC full dataset, 3 epochs',
    schedule_interval=None,
    catchup=False,
    tags=['phobert', 'sentiment', 'vietnamese', 'uc2'],
) as dag:

    t1 = PythonOperator(task_id='load_data',      python_callable=load_data)
    t2 = PythonOperator(task_id='preprocess',     python_callable=preprocess)
    t3 = PythonOperator(task_id='train_model',    python_callable=train_model)
    t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)

    t1 >> t2 >> t3 >> t4
