"""
UC2 PhoBERT Vietnamese Sentiment — MLflow Pipeline
Dataset: UIT-VSFC (Vietnamese Students Feedback Corpus)
Model: vinai/phobert-base (HuggingFace)

Chay tren GCP VM e2-medium:
    python3 -u train_uc2.py --mode all

Modes:
    all    = chay repeat (TC7) + config sweep (TC2)
    repeat = chi lap 3 lan de do TC7
    tc2    = chi chay config sweep
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlflow
import torch
import time
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

from shared.config_phobert import (
    PHOBERT_MODEL_NAME, NUM_LABELS, MAX_LENGTH,
    HPARAMS, DATASET_NAME, LABEL_COL, TEXT_COL,
    SAMPLE_SIZE, SEED, NUM_RUNS,
)
from shared.sampling_utils import sample_dataset

# CONFIG
CONFIG = {
    "tracking_uri":  "http://localhost:5000",
    "experiment":    "UC2_PhoBERT_MLflow",

    # TC2 — Config sweep (3 bo config khac nhau)
    "tc2_configs": [
        {"lr": 1e-5, "batch_size": 2, "epochs": 3},
        {"lr": 2e-5, "batch_size": 2, "epochs": 3},
        {"lr": 3e-5, "batch_size": 4, "epochs": 3},
    ],
}


# DATA LOADING & TOKENIZATION

def load_and_tokenize():
    """Load UIT-VSFC dataset, sample 500 mau, tokenize bang PhoBERT tokenizer."""
    print("  Loading dataset:", DATASET_NAME)

    # Doc parquet truc tiep tu HuggingFace (datasets v4.x bo ho tro dataset script)
    base = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default"
    urls = {
        "train":      f"{base}/train/0000.parquet",
        "validation": f"{base}/validation/0000.parquet",
        "test":       f"{base}/test/0000.parquet",
    }
    dataset = DatasetDict({
        split: Dataset.from_pandas(pd.read_parquet(url))
        for split, url in urls.items()
    })

    print(f"  Full data: Train={len(dataset['train'])}, Val={len(dataset['validation'])}, Test={len(dataset['test'])}")

    # Stratified sampling
    print(f"  Sampling {SAMPLE_SIZE} train samples (seed={SEED}, stratified)...")
    dataset = sample_dataset(dataset, LABEL_COL, SAMPLE_SIZE, SEED)
    print(f"  Sampled: Train={len(dataset['train'])}, Val={len(dataset['validation'])}, Test={len(dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(
            examples[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column(LABEL_COL, "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized, tokenizer


# TRAINING FUNCTION

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_one_run(tokenized, tokenizer, lr, batch_size, epochs, run_name):
    """Fine-tune PhoBERT 1 lan, log toan bo len MLflow."""

    grad_accum = HPARAMS["gradient_accumulation_steps"]
    use_grad_checkpoint = HPARAMS["gradient_checkpointing"]

    with mlflow.start_run(run_name=run_name):
        start_time = time.time()

        # ---- Log params ----
        mlflow.log_param("model", PHOBERT_MODEL_NAME)
        mlflow.log_param("dataset", DATASET_NAME)
        mlflow.log_param("sample_size", SAMPLE_SIZE)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("num_labels", NUM_LABELS)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("effective_batch_size", batch_size * grad_accum)
        mlflow.log_param("gradient_accumulation_steps", grad_accum)
        mlflow.log_param("gradient_checkpointing", use_grad_checkpoint)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("platform", "MLflow-GCP-VM")
        mlflow.log_param("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

        # ---- Load model ----
        model = AutoModelForSequenceClassification.from_pretrained(
            PHOBERT_MODEL_NAME,
            num_labels=NUM_LABELS,
        )

        if use_grad_checkpoint:
            model.gradient_checkpointing_enable()

        # ---- Training args ----
        training_args = TrainingArguments(
            output_dir="./results_uc2",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_total_limit=1,
            dataloader_num_workers=0,
        )

        # ---- Trainer ----
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )

        # ---- Train ----
        print(f"    Training {run_name}...")
        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start

        # ---- Evaluate on test set ----
        eval_start = time.time()
        test_results = trainer.evaluate(tokenized["test"])
        eval_time = time.time() - eval_start

        accuracy = test_results["eval_accuracy"]
        f1_macro = test_results["eval_f1_macro"]
        pipeline_time = time.time() - start_time

        # ---- Log metrics ----
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("eval_time_seconds", eval_time)
        mlflow.log_metric("pipeline_time_seconds", pipeline_time)

        print(f"    Accuracy={accuracy:.4f} | F1-macro={f1_macro:.4f} | Train={train_time:.1f}s | Eval={eval_time:.1f}s")
        return accuracy, f1_macro, pipeline_time


# MAIN

def main(mode="all"):
    mlflow.set_tracking_uri(CONFIG["tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment"])

    print("\n Loading & tokenizing UIT-VSFC dataset...")
    tokenized, tokenizer = load_and_tokenize()

    if mode in ("all", "repeat"):
        print(f"\n{'='*60}")
        print(f" PHAN 1: Chay lap {NUM_RUNS} lan — do TC7")
        print(f"{'='*60}")
        for i in range(1, NUM_RUNS + 1):
            print(f"\n  [PhoBERT] Lan {i}/{NUM_RUNS}")
            train_one_run(
                tokenized=tokenized,
                tokenizer=tokenizer,
                lr=HPARAMS["lr"],
                batch_size=HPARAMS["batch_size"],
                epochs=HPARAMS["epochs"],
                run_name=f"PhoBERT_run{i}",
            )

    if mode in ("all", "tc2"):
        print(f"\n{'='*60}")
        print(f" PHAN 2: TC2 Config Sweep ({len(CONFIG['tc2_configs'])} configs)")
        print(f"{'='*60}")
        for cfg in CONFIG["tc2_configs"]:
            run_name = f"TC2_lr{cfg['lr']}_batch{cfg['batch_size']}"
            print(f"\n  Config: {run_name}")
            train_one_run(
                tokenized=tokenized,
                tokenizer=tokenizer,
                run_name=run_name,
                **cfg,
            )

    print(f"\n{'='*60}")
    print(f" Hoan thanh UC2! Xem ket qua: {CONFIG['tracking_uri']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UC2 PhoBERT — MLflow Pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "repeat", "tc2"],
        default="all",
        help="all=chay tat ca | repeat=chi lap TC7 | tc2=chi config sweep",
    )
    args = parser.parse_args()
    main(args.mode)
