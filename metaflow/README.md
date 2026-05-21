# Metaflow — MLOps Evaluation

## Triển khai
- **UC1 (MNIST):** Chạy trên GCP VM (CPU) — local mode
- **UC2 (PhoBERT):** Chạy trên Google Colab (GPU T4) — local mode

## Cấu trúc
```
metaflow/
├── uc1_metaflow/
│   ├── train_uc1_metaflow.py    # Pipeline UC1
│   └── metaflow_uc1_results.csv # Kết quả
└── uc2_metaflow/
    ├── train_uc2_metaflow.py    # Pipeline UC2
    └── metaflow_uc2_results.csv # Kết quả
```

## Cách chạy
```bash
# UC1 (trên GCP VM)
cd metaflow/uc1_metaflow
python3 train_uc1_metaflow.py run --model SimpleNN
python3 train_uc1_metaflow.py run --model CNN

# UC2 (trên Colab)
# Mở notebook hoặc chạy:
python3 train_uc2_metaflow.py run --lr 2e-5
```
