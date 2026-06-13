# Metaflow — MLOps Thesis

Pipeline orchestration cho UC1 (MNIST) và UC2 (PhoBERT) dùng **Metaflow local mode** trên GCP VM e2-medium.

## Cấu trúc

```
metaflow/
├── uc1_metaflow/
│   └── train_uc1_metaflow.py   # Metaflow pipeline UC1 MNIST
├── uc2_metaflow/
│   └── train_uc2_metaflow.py   # Metaflow pipeline UC2 PhoBERT
├── run_uc1_all.sh              # Script chạy toàn bộ UC1 (TC7 + TC2)
├── run_uc2_all.sh              # Script chạy toàn bộ UC2 (TC7 + TC2)
└── view_results.py             # Xem kết quả các run
```

## Môi trường

| Item | Giá trị |
|---|---|
| Platform | GCP VM `e2-medium` (2 vCPU, 4GB RAM) |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| Metaflow | 2.10.0+ (local mode) |
| PyTorch | 2.11.0+cpu |
| Mode | Local datastore + local metadata |

## UC1 — MNIST Classification

**Model:** SimpleNN / DeepNN / CNN  
**Dataset:** MNIST (60,000 train / 10,000 test)  
**Hyperparams:** lr=0.001, batch_size=64, epochs=10, optimizer=Adam

### Chạy UC1

```bash
# Chạy toàn bộ (TC7 repeat + TC2 config sweep)
bash run_uc1_all.sh

# Chạy thủ công từng model
cd uc1_metaflow
python3 train_uc1_metaflow.py run
python3 train_uc1_metaflow.py run --model DeepNN
python3 train_uc1_metaflow.py run --model CNN
python3 train_uc1_metaflow.py run --lr 0.01 --batch_size 128
```

### Flow structure

```
start → train_and_evaluate → end
```

## UC2 — PhoBERT Vietnamese Sentiment

**Model:** vinai/phobert-base  
**Dataset:** UIT-VSFC — 500 mẫu stratified (seed=42)  
**Hyperparams:** lr=2e-5, batch_size=2, grad_accum=8, epochs=3  
**Effective batch size:** 2 × 8 = 16  
**Memory optimization:** gradient_checkpointing=True

### Chạy UC2

```bash
# Chạy toàn bộ (TC7 repeat + TC2 config sweep)
bash run_uc2_all.sh

# Chạy thủ công 1 run
cd uc2_metaflow
python3 train_uc2_metaflow.py run
python3 train_uc2_metaflow.py run --lr 1e-5
python3 train_uc2_metaflow.py run --lr 3e-5 --batch_size 4
```

### Flow structure

```
start → load_and_train → end
```

> **Lý do gộp load + train vào 1 step:** PyTorch model, DataLoader và HuggingFace Trainer/Dataset không thể pickle giữa các `@step` trong Metaflow local datastore.

## Xem kết quả

```bash
# Xem tất cả runs
python3 view_results.py

# Xem chi tiết 1 run (thay RUN_ID bằng ID thực tế)
python3 show_run_detail.py RUN_ID
```