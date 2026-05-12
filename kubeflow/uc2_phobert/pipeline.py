"""
UC2 — Vietnamese Sentiment Analysis — Kubeflow Pipelines 2.4.0
Model: vinai/phobert-base + Linear(768→3)
Dataset: UIT-VSFC full (train/val/test split chuẩn)
Pipeline: load_data >> preprocess >> train_model >> evaluate_model
Metrics: accuracy, F1-macro, train_time, eval_time

Fix: dùng JSON + torch.save thay pickle để truyền data giữa containers
"""

from kfp import dsl, compiler
from kfp.dsl import Output, Input, Metrics, Model, Dataset

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
    """Load UIT-VSFC, save dưới dạng JSON — không dùng pickle."""
    import time
    import json
    from datasets import load_dataset

    t0 = time.time()
    print("[load_data] Loading UIT-VSFC from HuggingFace...")

    dataset  = load_dataset("uitnlp/vietnamese_students_feedback")
    train_ds = dataset['train']
    val_ds   = dataset['validation']
    test_ds  = dataset['test']

    print(f"[load_data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    def save_json(ds, path):
        data = [{"sentence": row["sentence"], "sentiment": int(row["sentiment"])}
                for row in ds]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    save_json(train_ds, output_train.path)
    save_json(val_ds,   output_val.path)
    save_json(test_ds,  output_test.path)

    elapsed = round(time.time() - t0, 3)
    print(f"[load_data] done. time={elapsed}s")


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
    """Tokenize với PhoBERT tokenizer, save tensors dưới dạng .pt."""
    import time
    import json
    import torch
    from transformers import AutoTokenizer

    PHOBERT_MODEL_NAME = "vinai/phobert-base"
    MAX_LENGTH         = 256

    t0 = time.time()
    print("[preprocess] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    def load_and_tokenize(path, output_path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentences  = [d["sentence"] for d in data]
        sentiments = [d["sentiment"] for d in data]

        encoded = tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )

        result = {
            'input_ids':      encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels':         torch.tensor(sentiments, dtype=torch.long),
        }
        torch.save(result, output_path)
        print(f"  Tokenized {len(sentences)} samples")

    load_and_tokenize(input_train.path, output_train_tok.path)
    load_and_tokenize(input_val.path,   output_val_tok.path)
    load_and_tokenize(input_test.path,  output_test_tok.path)

    elapsed = round(time.time() - t0, 3)
    print(f"[preprocess] done. time={elapsed}s")


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
    """Fine-tune PhoBERT-base + Linear(768→3)."""
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from transformers import AutoModel
    from torch.optim import AdamW

    PHOBERT_MODEL_NAME = "vinai/phobert-base"
    NUM_LABELS         = 3
    LR                 = 2e-5
    BATCH_SIZE         = 8
    EPOCHS             = 3
    WEIGHT_DECAY       = 0.01

    train_data = torch.load(input_train_tok.path)
    val_data   = torch.load(input_val_tok.path)

    train_ds = TensorDataset(
        train_data['input_ids'],
        train_data['attention_mask'],
        train_data['labels'],
    )
    val_ds = TensorDataset(
        val_data['input_ids'],
        val_data['attention_mask'],
        val_data['labels'],
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)
            self.phobert.gradient_checkpointing_enable()

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    torch.manual_seed(42)
    device    = torch.device('cpu')
    model     = PhoBERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"[train] epochs={EPOCHS} lr={LR} batch={BATCH_SIZE} device=cpu")

    t0 = time.time()
    avg_loss = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels         = labels.to(device)
                val_loss      += criterion(model(input_ids, attention_mask), labels).item()

        print(f"  Epoch {epoch+1}/{EPOCHS} — train_loss: {avg_loss:.4f}, val_loss: {val_loss/len(val_loader):.4f}")

    train_time = round(time.time() - t0, 3)
    print(f"[train] done. train_time={train_time}s")

    torch.save(model.state_dict(), output_model.path)
    output_metrics.log_metric("train_time_seconds", train_time)
    output_metrics.log_metric("final_loss",         round(avg_loss, 4))


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
    """Evaluate trên test set, log accuracy + F1-macro."""
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from transformers import AutoModel
    from sklearn.metrics import accuracy_score, f1_score

    PHOBERT_MODEL_NAME = "vinai/phobert-base"
    NUM_LABELS         = 3
    BATCH_SIZE         = 8

    test_data = torch.load(input_test_tok.path)
    test_ds   = TensorDataset(
        test_data['input_ids'],
        test_data['attention_mask'],
        test_data['labels'],
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
            self.classifier = nn.Linear(768, NUM_LABELS)

        def forward(self, input_ids, attention_mask):
            out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    device = torch.device('cpu')
    model  = PhoBERTClassifier().to(device)
    model.load_state_dict(torch.load(input_model.path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            preds = model(input_ids.to(device), attention_mask.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    eval_time = round(time.time() - t0, 3)
    accuracy  = round(accuracy_score(all_labels, all_preds) * 100, 4)
    f1        = round(f1_score(all_labels, all_preds, average='macro') * 100, 4)

    print(f"[evaluate] Accuracy:  {accuracy:.4f}%")
    print(f"[evaluate] F1-macro:  {f1:.4f}%")
    print(f"[evaluate] Eval time: {eval_time}s")

    output_metrics.log_metric("accuracy",          accuracy)
    output_metrics.log_metric("f1_macro",          f1)
    output_metrics.log_metric("eval_time_seconds", eval_time)


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
    prep_task.set_memory_limit("8Gi")
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