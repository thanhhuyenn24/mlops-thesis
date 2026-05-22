"""
Shared PhoBERT fine-tuning config — dung chung cho ca 4 frameworks.
Khong sua file nay tru khi ca nhom dong y.
"""

# --- MODEL ---
PHOBERT_MODEL_NAME = "vinai/phobert-base"
NUM_LABELS = 3          # UIT-VSFC: 0=NEG, 1=NEU, 2=POS
MAX_LENGTH = 128        # Giam tu 256 de tiet kiem RAM tren e2-medium

# --- HYPERPARAMETERS ---
HPARAMS = {
    "lr":           2e-5,
    "batch_size":   2,          # Batch nho de vua RAM 4GB
    "epochs":       3,
    "optimizer":    "AdamW",
    "weight_decay": 0.01,
    "max_length":   MAX_LENGTH,
    "gradient_accumulation_steps": 8,   # Effective batch = 2 x 8 = 16
    "gradient_checkpointing": True,
}

# --- DATASET & SAMPLING ---
DATASET_NAME  = "uitnlp/vietnamese_students_feedback"
SAMPLE_SIZE   = 500     # Giam data de chay duoc tren e2-medium trong thoi gian hop ly
SEED          = 42      # Co dinh seed dam bao tat ca framework lay cung mau
STRATIFIED    = True    # Lay mau phan tang theo nhan NEG/NEU/POS

# --- EXPERIMENT ---
NUM_RUNS      = 3       # So lan chay de lay trung binh cho TC7
LABEL_COL     = "sentiment"
TEXT_COL      = "sentence"

# TC2 — config sweep
TC2_CONFIGS = [
    {"lr": 1e-5, "batch_size": 2, "epochs": 3},
    {"lr": 2e-5, "batch_size": 2, "epochs": 3},
    {"lr": 3e-5, "batch_size": 4, "epochs": 3},
]

# Label mapping
LABEL_MAP = {0: "NEG", 1: "NEU", 2: "POS"}
