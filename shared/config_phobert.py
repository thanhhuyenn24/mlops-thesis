"""
Shared PhoBERT fine-tuning config — dùng chung cho cả 4 frameworks.
Không sửa file này trừ khi cả nhóm đồng ý.
"""

# Model
PHOBERT_MODEL_NAME = "vinai/phobert-base"
NUM_LABELS = 3          # UIT-VSFC: 0=NEG, 1=NEU, 2=POS
MAX_LENGTH = 256

# Fine-tuning hyperparameters
HPARAMS = {
    "lr":           2e-5,
    "batch_size":   16,
    "epochs":       3,
    "optimizer":    "AdamW",
    "weight_decay": 0.01,
    "max_length":   MAX_LENGTH,
}

# Dataset
DATASET_NAME  = "uitnlp/vietnamese_students_feedback"
LABEL_COL     = "sentiment"
TEXT_COL      = "sentence"

# Label mapping — giữ nguyên theo UIT-VSFC split chuẩn
# 0: Negative, 1: Neutral, 2: Positive
LABEL_MAP = {0: "NEG", 1: "NEU", 2: "POS"}
