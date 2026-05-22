"""
UC2 — Vietnamese Sentiment Analysis — Apache Airflow
=====================================================
Model  : vinai/phobert-base + Linear(768 -> 3) classification head
Dataset: UIT-VSFC, stratified 500-sample train split, full val/test
runs   : 3 baseline repeats (TC7) + 3 hyperparameter sweep configs (TC2)

Pipeline per run: load_data >> preprocess >> train_model >> evaluate_model
"""

import sys
sys.path.insert(0, '/home/airflow/airflowServer/dags/shared')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_PATH = '/home/airflow/airflowServer/dags/shared'
TMP_DIR     = '/tmp/vsfc'

default_args = {
    'owner'     : 'airflow',
    'start_date': datetime(2024, 1, 1),
}

# ---------------------------------------------------------------------------
# Helper: build per-run task IDs
# ---------------------------------------------------------------------------

def _run_key(run_type: str, run_idx: int) -> str:
    """Return a unique task-group prefix, e.g. 'baseline_1' or 'sweep_2'."""
    return f"{run_type}_{run_idx}"


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def load_data(run_key: str, **kwargs):
    """
    Download UIT-VSFC from HuggingFace, apply stratified sampling to the
    train split (500 samples, SEED=42), and persist all three splits to disk.
    """
    import sys; sys.path.insert(0, SHARED_PATH)
    import os, pickle, time
    from datasets import load_dataset
    from config_phobert import DATASET_NAME, SAMPLE_SIZE, SEED, LABEL_COL
    from sampling_utils import stratified_sample

    t0 = time.time()
    dataset = load_dataset(DATASET_NAME)

    train_sampled = stratified_sample(
        dataset['train'], LABEL_COL, SAMPLE_SIZE, seed=SEED
    )

    os.makedirs(TMP_DIR, exist_ok=True)
    paths = {
        'train': f'{TMP_DIR}/{run_key}_train.pkl',
        'val'  : f'{TMP_DIR}/{run_key}_val.pkl',
        'test' : f'{TMP_DIR}/{run_key}_test.pkl',
    }
    with open(paths['train'], 'wb') as f: pickle.dump(train_sampled,      f)
    with open(paths['val'],   'wb') as f: pickle.dump(dataset['validation'], f)
    with open(paths['test'],  'wb') as f: pickle.dump(dataset['test'],       f)

    elapsed = round(time.time() - t0, 3)
    print(f"[load_data | {run_key}] train={len(train_sampled)} "
          f"val={len(dataset['validation'])} test={len(dataset['test'])} "
          f"time={elapsed}s")

    ti = kwargs['ti']
    ti.xcom_push(key='train_path', value=paths['train'])
    ti.xcom_push(key='val_path',   value=paths['val'])
    ti.xcom_push(key='test_path',  value=paths['test'])
    ti.xcom_push(key='load_time',  value=elapsed)


def preprocess(run_key: str, **kwargs):
    """
    Tokenize all three splits with the PhoBERT tokenizer and save the
    resulting torch-format datasets to disk.
    """
    import sys; sys.path.insert(0, SHARED_PATH)
    import pickle, time
    from transformers import AutoTokenizer
    from config_phobert import PHOBERT_MODEL_NAME, MAX_LENGTH, TEXT_COL, LABEL_COL

    ti = kwargs['ti']
    load_task_id = f'load_data_{run_key}'
    train_path = ti.xcom_pull(key='train_path', task_ids=load_task_id)
    val_path   = ti.xcom_pull(key='val_path',   task_ids=load_task_id)
    test_path  = ti.xcom_pull(key='test_path',  task_ids=load_task_id)

    t0        = time.time()
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    def _tokenize(split):
        tokenized = split.map(
            lambda x: tokenizer(
                x[TEXT_COL],
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
        )
        keep = ['input_ids', 'attention_mask', LABEL_COL]
        tokenized = tokenized.remove_columns(
            [c for c in tokenized.column_names if c not in keep]
        )
        tokenized.set_format(type='torch', columns=keep)
        return tokenized

    with open(train_path, 'rb') as f: train_ds = pickle.load(f)
    with open(val_path,   'rb') as f: val_ds   = pickle.load(f)
    with open(test_path,  'rb') as f: test_ds  = pickle.load(f)

    tok_paths = {
        'train': f'{TMP_DIR}/{run_key}_train_tok.pkl',
        'val'  : f'{TMP_DIR}/{run_key}_val_tok.pkl',
        'test' : f'{TMP_DIR}/{run_key}_test_tok.pkl',
    }
    with open(tok_paths['train'], 'wb') as f: pickle.dump(_tokenize(train_ds), f)
    with open(tok_paths['val'],   'wb') as f: pickle.dump(_tokenize(val_ds),   f)
    with open(tok_paths['test'],  'wb') as f: pickle.dump(_tokenize(test_ds),  f)

    elapsed = round(time.time() - t0, 3)
    print(f"[preprocess | {run_key}] done. time={elapsed}s")

    ti.xcom_push(key='tok_train_path',  value=tok_paths['train'])
    ti.xcom_push(key='tok_val_path',    value=tok_paths['val'])
    ti.xcom_push(key='tok_test_path',   value=tok_paths['test'])
    ti.xcom_push(key='preprocess_time', value=elapsed)


def train_model(run_key: str, hparams: dict, **kwargs):
    """
    Fine-tune vinai/phobert-base with a linear classification head.
    Supports gradient accumulation and optional gradient checkpointing.
    Model weights are saved to /tmp/vsfc for downstream evaluation.
    """
    import sys; sys.path.insert(0, SHARED_PATH)
    import pickle, time
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from torch.optim import AdamW
    from config_phobert import PHOBERT_MODEL_NAME, NUM_LABELS, SEED

    ti           = kwargs['ti']
    preproc_id   = f'preprocess_{run_key}'
    train_path   = ti.xcom_pull(key='tok_train_path', task_ids=preproc_id)
    val_path     = ti.xcom_pull(key='tok_val_path',   task_ids=preproc_id)

    LR           = hparams['lr']
    BATCH_SIZE   = hparams['batch_size']
    EPOCHS       = hparams['epochs']
    WEIGHT_DECAY = hparams.get('weight_decay', 0.01)
    ACCUM_STEPS  = hparams.get('gradient_accumulation_steps', 8)
    LABEL_COL    = 'sentiment'

    with open(train_path, 'rb') as f: train_ds = pickle.load(f)
    with open(val_path,   'rb') as f: val_ds   = pickle.load(f)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)
            if hparams.get('gradient_checkpointing'):
                self.encoder.gradient_checkpointing_enable()

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    torch.manual_seed(SEED)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = PhoBERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"[train | {run_key}] device={device} lr={LR} "
          f"batch={BATCH_SIZE} accum={ACCUM_STEPS} epochs={EPOCHS}")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch[LABEL_COL].to(device)

            loss = criterion(model(input_ids, attention_mask), labels) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits    = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                )
                val_loss += criterion(logits, batch[LABEL_COL].to(device)).item()

        print(f"  Epoch {epoch + 1}/{EPOCHS} — "
              f"train_loss={total_loss / len(train_loader):.4f} "
              f"val_loss={val_loss / len(val_loader):.4f}")

    train_time = round(time.time() - t0, 3)
    print(f"[train | {run_key}] done. train_time={train_time}s")

    model_path = f'{TMP_DIR}/{run_key}_phobert.pth'
    torch.save(model.state_dict(), model_path)

    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='train_time', value=train_time)


def evaluate_model(run_key: str, hparams: dict, **kwargs):
    """
    Evaluate the fine-tuned model on the full test set.
    Logs accuracy, F1-macro, and all timing breakdowns required by TC7.
    """
    import sys; sys.path.insert(0, SHARED_PATH)
    import pickle, time
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from sklearn.metrics import accuracy_score, f1_score
    from config_phobert import PHOBERT_MODEL_NAME, NUM_LABELS

    ti           = kwargs['ti']
    preproc_id   = f'preprocess_{run_key}'
    train_id     = f'train_model_{run_key}'
    load_id      = f'load_data_{run_key}'

    test_path       = ti.xcom_pull(key='tok_test_path',   task_ids=preproc_id)
    model_path      = ti.xcom_pull(key='model_path',      task_ids=train_id)
    train_time      = ti.xcom_pull(key='train_time',      task_ids=train_id)
    load_time       = ti.xcom_pull(key='load_time',       task_ids=load_id)
    preprocess_time = ti.xcom_pull(key='preprocess_time', task_ids=preproc_id)

    BATCH_SIZE = hparams['batch_size']
    LABEL_COL  = 'sentiment'

    with open(test_path, 'rb') as f: test_ds = pickle.load(f)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PhoBERTClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            preds = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
            ).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(batch[LABEL_COL].numpy())

    eval_time     = round(time.time() - t0, 3)
    accuracy      = round(accuracy_score(all_labels, all_preds) * 100, 4)
    f1            = round(f1_score(all_labels, all_preds, average='macro') * 100, 4)
    pipeline_time = round(load_time + preprocess_time + train_time + eval_time, 3)

    print(f"[evaluate | {run_key}] Results:")
    print(f"  lr             : {hparams['lr']}")
    print(f"  batch_size     : {hparams['batch_size']}")
    print(f"  Accuracy       : {accuracy:.4f}%")
    print(f"  F1-macro       : {f1:.4f}%")
    print(f"  Load time      : {load_time}s")
    print(f"  Preprocess time: {preprocess_time}s")
    print(f"  Train time     : {train_time}s")
    print(f"  Eval time      : {eval_time}s")
    print(f"  Pipeline time  : {pipeline_time}s")

    ti.xcom_push(key='accuracy',       value=accuracy)
    ti.xcom_push(key='f1',             value=f1)
    ti.xcom_push(key='eval_time',      value=eval_time)
    ti.xcom_push(key='pipeline_time',  value=pipeline_time)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

# Run configurations
# - baseline: 3 repeats with default hyperparameters (used for TC7)
# - sweep   : 3 configs with varying lr (used for TC2)

from config_phobert import HPARAMS as DEFAULT_HPARAMS
from config_phobert import TC2_CONFIGS

RUN_GROUPS = (
    [('baseline', i + 1, DEFAULT_HPARAMS)       for i in range(3)] +
    [('sweep',    i + 1, TC2_CONFIGS[i])         for i in range(3)]
)

with DAG(
    dag_id          ='vietnamese_sentiment_airflow',
    default_args    =default_args,
    description     ='UC2 PhoBERT — 3 baseline runs (TC7) + 3 sweep runs (TC2)',
    schedule_interval=None,
    catchup         =False,
    tags            =['phobert', 'sentiment', 'uc2', 'tc2', 'tc7'],
) as dag:

    for run_type, run_idx, hparams in RUN_GROUPS:
        rk = _run_key(run_type, run_idx)

        t_load = PythonOperator(
            task_id         =f'load_data_{rk}',
            python_callable =load_data,
            op_kwargs       ={'run_key': rk},
        )
        t_prep = PythonOperator(
            task_id         =f'preprocess_{rk}',
            python_callable =preprocess,
            op_kwargs       ={'run_key': rk},
        )
        t_train = PythonOperator(
            task_id         =f'train_model_{rk}',
            python_callable =train_model,
            op_kwargs       ={'run_key': rk, 'hparams': hparams},
        )
        t_eval = PythonOperator(
            task_id         =f'evaluate_model_{rk}',
            python_callable =evaluate_model,
            op_kwargs       ={'run_key': rk, 'hparams': hparams},
        )

        t_load >> t_prep >> t_train >> t_eval