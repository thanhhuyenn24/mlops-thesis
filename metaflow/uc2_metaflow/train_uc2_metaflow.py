"""
UC2 PhoBERT Vietnamese Sentiment — Metaflow Pipeline (Local Mode)

Chay tren GCP VM e2-medium:
  python3 train_uc2_metaflow.py run
  python3 train_uc2_metaflow.py run --lr 1e-5
  python3 train_uc2_metaflow.py run --lr 3e-5 --batch_size 4
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

from metaflow import FlowSpec, step, Parameter
import time


class PhoBERTSentimentFlow(FlowSpec):
    """Pipeline UC2 PhoBERT Sentiment — Metaflow"""

    lr = Parameter('lr', default=2e-5, type=float,
                   help='Learning rate')
    batch_size = Parameter('batch_size', default=2, type=int,
                           help='Batch size (effective = batch_size x grad_accum)')
    epochs = Parameter('epochs', default=3, type=int,
                       help='Number of epochs')
    max_length = Parameter('max_length', default=128, type=int,
                           help='Max token length for PhoBERT')
    grad_accum = Parameter('grad_accum', default=8, type=int,
                           help='Gradient accumulation steps')

    @step
    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*50}")
        print(f"  UC2 PhoBERT Sentiment — Metaflow (GCP VM)")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, "
              f"grad_accum={self.grad_accum}, epochs={self.epochs}")
        print(f"  Effective batch size: {self.batch_size * self.grad_accum}")
        print(f"  Optimizer: AdamW")
        print(f"{'='*50}")
        self.next(self.load_and_train)

    @step
    def load_and_train(self):
        """
        Load data + Tokenize + Train + Evaluate (gop 1 step)

        LY DO GOP: PyTorch models, DataLoaders, va HuggingFace Datasets
        KHONG the pickle giua cac @step trong Metaflow.
        """
        import numpy as np
        import torch
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
            PHOBERT_MODEL_NAME, NUM_LABELS, DATASET_NAME, TEXT_COL, LABEL_COL,
            SAMPLE_SIZE, SEED,
        )
        from shared.sampling_utils import sample_dataset

        # ========== LOAD DATASET ==========
        print("  Loading UIT-VSFC dataset...")
        urls = {
            "train": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "validation": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
            "test": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        }

        train_df = pd.read_parquet(urls["train"])
        val_df = pd.read_parquet(urls["validation"])
        test_df = pd.read_parquet(urls["test"])

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        })
        print(f"  Full data: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")

        # ========== STRATIFIED SAMPLING ==========
        print(f"  Sampling {SAMPLE_SIZE} train samples (seed={SEED}, stratified)...")
        dataset = sample_dataset(dataset, LABEL_COL, SAMPLE_SIZE, SEED)
        print(f"  Sampled: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")

        # ========== TOKENIZE ==========
        print("  Tokenizing with PhoBERT...")
        tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

        def tokenize_fn(examples):
            return tokenizer(
                examples[TEXT_COL],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized = tokenized.rename_column(LABEL_COL, "labels")
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # ========== TRAIN ==========
        print("  Fine-tuning PhoBERT...")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_score(labels, preds, average="macro"),
            }

        model = AutoModelForSequenceClassification.from_pretrained(
            PHOBERT_MODEL_NAME, num_labels=NUM_LABELS
        )
        model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            output_dir="./results_uc2_metaflow",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.lr,
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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )

        train_start = time.time()
        trainer.train()
        self.train_time = time.time() - train_start

        # ========== EVALUATE ==========
        print("  Evaluating on test set...")
        eval_start = time.time()
        test_results = trainer.evaluate(tokenized["test"])
        self.eval_time = time.time() - eval_start

        self.accuracy = test_results["eval_accuracy"]
        self.f1_macro = test_results["eval_f1_macro"]
        self.pipeline_time = time.time() - self.start_time
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        print(f"\n  Accuracy: {self.accuracy:.4f}")
        print(f"  F1-macro: {self.f1_macro:.4f}")
        print(f"  Train time: {self.train_time:.1f}s")
        print(f"  Eval time: {self.eval_time:.1f}s")
        print(f"  Pipeline time: {self.pipeline_time:.1f}s")
        print(f"  Device: {self.gpu_name}")
        self.next(self.end)

    @step
    def end(self):
        print(f"\n{'='*50}")
        print(f"  FINAL RESULT")
        print(f"  Model: vinai/phobert-base")
        print(f"  Dataset: UIT-VSFC (500 samples stratified)")
        print(f"  Accuracy: {self.accuracy:.4f}")
        print(f"  F1-macro: {self.f1_macro:.4f}")
        print(f"  Train time: {self.train_time:.1f}s")
        print(f"  Eval time: {self.eval_time:.1f}s")
        print(f"  Pipeline time: {self.pipeline_time:.1f}s")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, "
              f"grad_accum={self.grad_accum}, epochs={self.epochs}, optimizer=AdamW")
        print(f"  Platform: Metaflow Local Mode (GCP VM {self.gpu_name})")
        print(f"{'='*50}")


if __name__ == '__main__':
    PhoBERTSentimentFlow()
