"""
Kubeflow Pipelines — UC2: Vietnamese Sentiment Analysis (PhoBERT)
6 runs: 3 baseline + 3 TC2 sweep. 500 stratified samples, SEED=42.
"""

from kfp import dsl
from kfp.client import Client

PHOBERT_IMAGE = "localhost:32000/kfp-phobert:latest"

@dsl.component(base_image=PHOBERT_IMAGE)
def traineval(
    lr            : float,
    batch_size    : int,
    epochs        : int,
    run_tag       : str,
    training_seed : int,
) -> str:
    import json, time, requests, os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    from sklearn.metrics import f1_score

    # ── Config ────────────────────────────────────────────────────────────────
    PHOBERT_MODEL_NAME     = "vinai/phobert-base"
    NUM_LABELS             = 3
    MAX_LENGTH             = 128
    WEIGHT_DECAY           = 0.01
    ACCUMULATION_STEPS     = 8
    GRADIENT_CHECKPOINTING = True
    SAMPLE_SIZE            = 500
    SEED                   = 42

    # Cache model locally to avoid re-download on retry
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
    os.makedirs("/tmp/hf_cache", exist_ok=True)

    # ── Download dataset with retry + local cache ─────────────────────────────
    base    = "https://datasets-server.huggingface.co/rows"
    dataset = "uitnlp/vietnamese_students_feedback"

    def fetch_with_retry(url, params, max_retries=5):
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=60)
                if resp.status_code == 200 and resp.text.strip():
                    return resp.json()
                print(f"[WARN] attempt {attempt+1}: status={resp.status_code}")
            except Exception as e:
                print(f"[WARN] attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to fetch after {max_retries} attempts: {params}")

    def fetch_split(split, limit=10000):
        # Check local cache first
        cache_file = f"/tmp/hf_cache/{split}_{limit}.json"
        if os.path.exists(cache_file):
            print(f"[INFO] Loading {split} from cache...")
            with open(cache_file) as f:
                data = json.load(f)
            return data["sentences"], data["labels"]

        sentences, labels = [], []
        offset = 0
        while True:
            data = fetch_with_retry(base, {
                "dataset": dataset, "config": "default",
                "split": split, "offset": offset, "limit": 100,
            })
            rows = data.get("rows", [])
            if not rows:
                break
            for r in rows:
                sentences.append(r["row"]["sentence"])
                labels.append(r["row"]["sentiment"])
            offset += len(rows)
            if offset >= limit:
                break

        # Save to cache
        with open(cache_file, "w") as f:
            json.dump({"sentences": sentences, "labels": labels}, f)
        return sentences, labels

    def stratified_sample(sentences, labels, n, seed=42):
        labels_arr    = np.array(labels)
        unique_labels = np.unique(labels_arr)
        rng           = np.random.default_rng(seed)
        selected      = []
        n_per_label   = n // len(unique_labels)
        remainder     = n % len(unique_labels)
        for i, label in enumerate(unique_labels):
            idxs   = np.where(labels_arr == label)[0]
            n_take = min(n_per_label + (1 if i < remainder else 0), len(idxs))
            selected.extend(rng.choice(idxs, size=n_take, replace=False).tolist())
        rng.shuffle(selected)
        return selected

    print(f"[INFO] {run_tag} — downloading data...")
    train_s, train_l = fetch_split("train")
    test_s,  test_l  = fetch_split("test", limit=2000)

    idx     = stratified_sample(train_s, train_l, SAMPLE_SIZE, SEED)
    train_s = [train_s[i] for i in idx]
    train_l = [train_l[i] for i in idx]
    print(f"[INFO] Train: {len(train_s)}, Test: {len(test_s)}")

    # ── Load model with retry ─────────────────────────────────────────────────
    def load_with_retry(loader_fn, max_retries=3):
        for attempt in range(max_retries):
            try:
                return loader_fn()
            except Exception as e:
                print(f"[WARN] model load attempt {attempt+1}: {e}")
                time.sleep(5)
        raise RuntimeError("Failed to load model after retries")

    torch.manual_seed(training_seed)
    device    = torch.device("cpu")
    tokenizer = load_with_retry(
        lambda: AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    )

    # ── Tokenise ──────────────────────────────────────────────────────────────
    class SentimentDataset(TorchDataset):
        def __init__(self, sentences, labels):
            enc = tokenizer(
                sentences, padding="max_length", truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels         = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return {
                "input_ids"      : self.input_ids[idx],
                "attention_mask" : self.attention_mask[idx],
                "labels"         : self.labels[idx],
            }

    train_loader = DataLoader(
        SentimentDataset(train_s, train_l), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        SentimentDataset(test_s, test_l), batch_size=8, shuffle=False
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_with_retry(
        lambda: AutoModelForSequenceClassification.from_pretrained(
            PHOBERT_MODEL_NAME, num_labels=NUM_LABELS
        ).to(device)
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, (len(train_loader) * epochs) // ACCUMULATION_STEPS)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = max(1, int(0.1 * total_steps)),
        num_training_steps = total_steps,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    pipeline_start = time.time()
    train_start    = time.time()
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["labels"].to(device),
            )
            (outputs.loss / ACCUMULATION_STEPS).backward()
            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    train_time = time.time() - train_start

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_start = time.time()
    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            preds   = outputs.logits.argmax(dim=1)
            labels  = batch["labels"].to(device)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_time     = time.time() - eval_start
    pipeline_time = time.time() - pipeline_start

    result = {
        "run_id"          : run_tag,
        "lr"              : lr,
        "batch_size"      : batch_size,
        "epochs"          : epochs,
        "training_seed"   : training_seed,
        "accuracy"        : round(correct / total, 6),
        "f1_macro"        : round(f1_score(all_labels, all_preds, average="macro"), 6),
        "train_time_s"    : round(train_time, 4),
        "eval_time_s"     : round(eval_time, 4),
        "pipeline_time_s" : round(pipeline_time, 4),
    }
    print("CSV_ROW:" + ",".join(str(result[k]) for k in [
        "run_id", "lr", "batch_size", "training_seed",
        "accuracy", "f1_macro", "train_time_s", "eval_time_s", "pipeline_time_s",
    ]))
    print(json.dumps(result, indent=2))
    return json.dumps(result)


@dsl.pipeline(
    name        = "Vietnamese Sentiment — KFP UC2",
    description = "6 runs: 3 baseline + 3 TC2 sweep. 500 stratified samples, SEED=42.",
)
def sentiment_pipeline():
    HPARAMS     = {"lr": 2e-5, "batch_size": 2, "epochs": 3}
    NUM_RUNS    = 3
    TC2_CONFIGS = [
        {"lr": 1e-5, "batch_size": 2, "epochs": 3},
        {"lr": 2e-5, "batch_size": 2, "epochs": 3},
        {"lr": 3e-5, "batch_size": 4, "epochs": 3},
    ]

    prev_task = None

    BASELINE_SEEDS = [43, 44, 45]

    for run_idx in range(1, NUM_RUNS + 1):
        run_tag = f"baseline_run{run_idx}"
        task = traineval(
            lr=HPARAMS["lr"],
            batch_size=HPARAMS["batch_size"],
            epochs=HPARAMS["epochs"],
            run_tag=run_tag,
            training_seed=BASELINE_SEEDS[run_idx - 1],
        )
        task.set_caching_options(enable_caching=False)
        task.set_memory_limit("8G")
        task.set_cpu_limit("2")
        task.set_display_name(run_tag)
        if prev_task is not None:
            task.after(prev_task)
        prev_task = task

    for sweep_idx, cfg in enumerate(TC2_CONFIGS, start=1):
        run_tag = f"sweep{sweep_idx}"
        task = traineval(
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            run_tag=run_tag,
            training_seed=42,
        )
        task.set_caching_options(enable_caching=False)
        task.set_memory_limit("8G")
        task.set_cpu_limit("2")
        task.set_display_name(run_tag)
        task.after(prev_task)
        prev_task = task


if __name__ == "__main__":
    import kfp
    import os

    PIPELINE_YAML = "sentiment_pipeline.yaml"
    KFP_HOST      = "http://localhost:8888"

    kfp.compiler.Compiler().compile(sentiment_pipeline, PIPELINE_YAML)
    print(f"[INFO] Compiled → {PIPELINE_YAML}")

    if os.getenv("GITHUB_ACTIONS"):
        raise SystemExit(0)

    client = Client(host=KFP_HOST)
    run = client.create_run_from_pipeline_func(
        pipeline_func  = sentiment_pipeline,
        run_name       = "UC2-Sentiment-6runs",
        enable_caching = False,
    )
    print(f"[INFO] Submitted → run_id: {run.run_id}")
