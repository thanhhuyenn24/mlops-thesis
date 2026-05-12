"""
UC2 — Vietnamese Sentiment Analysis — Kubeflow Pipelines 2.4.0
Model: vinai/phobert-base + Linear(768→3)
Dataset: UIT-VSFC full (train/val/test split chuẩn)
Pipeline: load_data >> preprocess >> train_model >> evaluate_model
Metrics: accuracy, F1-macro, train_time, eval_time
"""

from kfp import dsl, compiler
from kfp.dsl import Output, Input, Metrics, Model, Dataset

# ── Packages ─────────────────────────────────────────────────────────────────
BASE_PACKAGES = [
    "numpy<2.0",
    "torch==2.2.2+cpu",
    "transformers==4.44.0",
    "datasets==2.19.0",
    "scikit-learn==1.4.2",
    "pyarrow==14.0.2",
]
PIP_URLS = [
    "https://download.pytorch.org/whl/cpu",
    "https://pypi.org/simple",
]

# ── Config — đồng nhất với Airflow/MLflow/Metaflow ───────────────────────────
PHOBERT_MODEL_NAME = "vinai/phobert-base"
NUM_LABELS         = 3
MAX_LENGTH         = 256
LR                 = 2e-5
BATCH_SIZE         = 8
EPOCHS             = 3
WEIGHT_DECAY       = 0.01
DATASET_NAME       = "uitnlp/vietnamese_students_feedback"
TEXT_COL           = "sentence"
LABEL_COL          = "sentiment"


# ── Step 1: Load data ────────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=BASE_PACKAGES,
    pip_index_urls=PIP_URLS,
)
def load_data(
    output_train: Output[Dataset],
    output_val:   Output[Dataset],
    output_test:  Output[Dataset],
):
    """Load UIT-VSFC từ HuggingFace, dùng split chuẩn."""
    import time
    import pickle
    from datasets import load_dataset

    cache_dir = "/root/.cache/huggingface/datasets/uitnlp___vietnamese_students_feedback"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[load_data] Cleared old cache at {cache_dir}")

    t0 = time.time()
    print("[load_data] Loading UIT-VSFC...")

    dataset = load_dataset("uitnlp/vietnamese_students_feedback", download_mode="force_redownload")
    train_ds = dataset['train']
    val_ds   = dataset['validation']
    test_ds  = dataset['test']

    print(f"[load_data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    with open(output_train.path, 'wb') as f: pickle.dump(train_ds, f)
    with open(output_val.path,   'wb') as f: pickle.dump(val_ds,   f)
    with open(output_test.path,  'wb') as f: pickle.dump(test_ds,  f)

    elapsed = round(time.time() - t0, 3)
    print(f"[load_data] done. time={elapsed}s")


# ── Step 2: Preprocess ───────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=BASE_PACKAGES,
    pip_index_urls=PIP_URLS,
)
def preprocess(
    input_train: Input[Dataset],
    input_val:   Input[Dataset],
    input_test:  Input[Dataset],
    output_train_tok: Output[Dataset],
    output_val_tok:   Output[Dataset],
    output_test_tok:  Output[Dataset],
):
    """Tokenize UIT-VSFC với PhoBERT tokenizer."""
    import time
    import pickle
    from transformers import AutoTokenizer

    PHOBERT_MODEL_NAME = "vinai/phobert-base"
    MAX_LENGTH         = 256
    TEXT_COL           = "sentence"
    LABEL_COL          = "sentiment"

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

    with open(input_train.path, 'rb') as f: train_ds = pickle.load(f)
    with open(input_val.path,   'rb') as f: val_ds   = pickle.load(f)
    with open(input_test.path,  'rb') as f: test_ds  = pickle.load(f)

    print("[preprocess] Tokenizing...")
    train_tok = tokenize(train_ds)
    val_tok   = tokenize(val_ds)
    test_tok  = tokenize(test_ds)

    with open(output_train_tok.path, 'wb') as f: pickle.dump(train_tok, f)
    with open(output_val_tok.path,   'wb') as f: pickle.dump(val_tok,   f)
    with open(output_test_tok.path,  'wb') as f: pickle.dump(test_tok,  f)

    elapsed = round(time.time() - t0, 3)
    print(f"[preprocess] done. time={elapsed}s")


# ── Step 3: Train model ──────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=BASE_PACKAGES,
    pip_index_urls=PIP_URLS,
)
def train_model(
    input_train_tok: Input[Dataset],
    input_val_tok:   Input[Dataset],
    output_model:    Output[Model],
    output_metrics:  Output[Metrics],
):
    """Fine-tune PhoBERT-base + Linear(768→3) trên UIT-VSFC train set."""
    import time
    import pickle
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from torch.optim import AdamW

    PHOBERT_MODEL_NAME = "vinai/phobert-base"
    NUM_LABELS         = 3
    LR                 = 2e-5
    BATCH_SIZE         = 8
    EPOCHS             = 3
    WEIGHT_DECAY       = 0.01
    LABEL_COL          = "sentiment"

    with open(input_train_tok.path, 'rb') as f: train_ds = pickle.load(f)
    with open(input_val_tok.path,   'rb') as f: val_ds   = pickle.load(f)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)
            # Gradient checkpointing để tiết kiệm RAM
            self.phobert.gradient_checkpointing_enable()

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls)

    device    = torch.device('cpu')
    model     = PhoBERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"[train] Fine-tuning PhoBERT — epochs={EPOCHS} lr={LR} batch={BATCH_SIZE} device=cpu")

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

        # Validation
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

    torch.save(model.state_dict(), output_model.path)
    output_metrics.log_metric("train_time_seconds", train_time)
    output_metrics.log_metric("final_loss",         round(avg_loss, 4))


# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=BASE_PACKAGES,
    pip_index_urls=PIP_URLS,
)
def evaluate_model(
    input_test_tok: Input[Dataset],
    input_model:    Input[Model],
    output_metrics: Output[Metrics],
):
    """Evaluate PhoBERT trên test set, log accuracy + F1-macro."""
    import time
    import pickle
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from sklearn.metrics import accuracy_score, f1_score

    PHOBERT_MODEL_NAME = "vinai/phobert-base"
    NUM_LABELS         = 3
    BATCH_SIZE         = 8
    LABEL_COL          = "sentiment"

    with open(input_test_tok.path, 'rb') as f: test_ds = pickle.load(f)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls)

    device = torch.device('cpu')
    model  = PhoBERTClassifier().to(device)
    model.load_state_dict(torch.load(input_model.path, map_location=device))
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

    print(f"[evaluate] Accuracy: {accuracy:.4f}%")
    print(f"[evaluate] F1-macro: {f1:.4f}%")
    print(f"[evaluate] Eval time: {eval_time}s")

    output_metrics.log_metric("accuracy",          accuracy)
    output_metrics.log_metric("f1_macro",          f1)
    output_metrics.log_metric("eval_time_seconds", eval_time)


# ── Pipeline definition ───────────────────────────────────────────────────────
@dsl.pipeline(
    name="phobert-sentiment-kubeflow",
    description="UC2 PhoBERT fine-tune — UIT-VSFC full, 3 epochs, CPU",
)
def phobert_pipeline():
    load_task = load_data()
    load_task.set_memory_limit("4Gi")
    load_task.set_cpu_limit("2")
    load_task.set_caching_options(enable_caching=False)

    prep_task = preprocess(
        input_train=load_task.outputs["output_train"],
        input_val=load_task.outputs["output_val"],
        input_test=load_task.outputs["output_test"],
    )
    prep_task.set_memory_limit("4Gi")
    prep_task.set_cpu_limit("2")
    prep_task.set_caching_options(enable_caching=False)
    prep_task.after(load_task)

    train_task = train_model(
        input_train_tok=prep_task.outputs["output_train_tok"],
        input_val_tok=prep_task.outputs["output_val_tok"],
    )
    train_task.set_memory_limit("12Gi")
    train_task.set_cpu_limit("4")
    train_task.set_caching_options(enable_caching=False)
    train_task.after(prep_task)

    eval_task = evaluate_model(
        input_test_tok=prep_task.outputs["output_test_tok"],
        input_model=train_task.outputs["output_model"],
    )
    eval_task.set_memory_limit("8Gi")
    eval_task.set_cpu_limit("2")
    eval_task.set_caching_options(enable_caching=False)
    eval_task.after(train_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=phobert_pipeline,
        package_path="phobert_pipeline.yaml",
    )
    print("Compiled: phobert_pipeline.yaml")