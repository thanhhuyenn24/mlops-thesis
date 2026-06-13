#!/bin/bash
# ============================================================
# Script chay toan bo UC2 tren Metaflow (GCP VM e2-medium)
# Chay: bash run_uc2_all.sh
# ============================================================

set -e
cd "$(dirname "$0")/uc2_metaflow"

echo "=========================================="
echo " UC2 PhoBERT — Metaflow (GCP VM e2-medium)"
echo " Config: lr=2e-5, batch=2, grad_accum=8"
echo " Data: 500 samples stratified (seed=42)"
echo "=========================================="

# --- PHAN 1: Repeat 3 lan (TC7) ---
echo ""
echo ">>> PhoBERT x 3 runs (TC7)"
python3 -u train_uc2_metaflow.py run
python3 -u train_uc2_metaflow.py run
python3 -u train_uc2_metaflow.py run

# --- PHAN 2: TC2 Config Sweep ---
echo ""
echo ">>> TC2 Config Sweep (3 configs)"
python3 -u train_uc2_metaflow.py run --lr 1e-5 --batch_size 2
python3 -u train_uc2_metaflow.py run --lr 2e-5 --batch_size 2
python3 -u train_uc2_metaflow.py run --lr 3e-5 --batch_size 4

# --- Xem ket qua ---
echo ""
echo ">>> Ket qua:"
cd ..
python3 -u view_results.py

echo ""
echo "=========================================="
echo " UC2 Metaflow HOAN THANH!"
echo "=========================================="
