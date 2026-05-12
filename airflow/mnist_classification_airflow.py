"""
UC1 — MNIST Classification — Apache Airflow
Pipeline: load_data >> train_model >> evaluate_model
Chạy 3 lần × 3 models = 9 runs, log accuracy + F1 + timing
"""

import sys
import os
sys.path.insert(0, '/opt/airflow/dags/shared')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

# ── Hyperparameters & config từ shared file ──────────────────────────────────
LR         = 0.001
BATCH_SIZE = 64
EPOCHS     = 10
MODELS     = ["SimpleNN", "DeepNN", "CNN"]
NUM_RUNS   = 3   # mỗi model chạy 3 lần → 9 runs tổng


# ── Step 1: Load data ────────────────────────────────────────────────────────
def load_data(**kwargs):
    import time
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import pickle

    t0 = time.time()
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root='/tmp/mnist_data', train=True,
                               download=True, transform=transform)
    test_ds  = datasets.MNIST(root='/tmp/mnist_data', train=False,
                               download=True, transform=transform)

    # Lưu đường dẫn dataset, không truyền DataLoader qua XCom
    elapsed = time.time() - t0
    print(f"[load_data] MNIST downloaded. Train: {len(train_ds)}, Test: {len(test_ds)}, Time: {elapsed:.2f}s")

    kwargs['ti'].xcom_push(key='data_root',   value='/tmp/mnist_data')
    kwargs['ti'].xcom_push(key='load_time_s', value=round(elapsed, 3))


# ── Step 2: Train model ──────────────────────────────────────────────────────
def train_model(**kwargs):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import time
    import json

    # Import shared model definitions
    from models_mnist import MODEL_CLASSES

    ti        = kwargs['ti']
    model_name = kwargs['model_name']
    run_id     = kwargs['run_id']
    data_root  = ti.xcom_pull(key='data_root', task_ids='load_data')

    transform    = transforms.Compose([transforms.ToTensor()])
    train_ds     = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = MODEL_CLASSES[model_name]()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"[train] model={model_name} run={run_id} epochs={EPOCHS} lr={LR} batch={BATCH_SIZE}")

    t0 = time.time()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}")

    train_time = round(time.time() - t0, 3)
    print(f"[train] done. train_time={train_time}s")

    model_path = f'/tmp/mnist_{model_name}_run{run_id}.pth'
    torch.save(model.state_dict(), model_path)

    xcom_key = f'{model_name}_run{run_id}'
    ti.xcom_push(key=f'{xcom_key}_model_path',  value=model_path)
    ti.xcom_push(key=f'{xcom_key}_train_time',  value=train_time)
    ti.xcom_push(key=f'{xcom_key}_loss_history', value=json.dumps(history))


# ── Step 3: Evaluate model ───────────────────────────────────────────────────
def evaluate_model(**kwargs):
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, f1_score
    import time

    from models_mnist import MODEL_CLASSES

    ti         = kwargs['ti']
    model_name = kwargs['model_name']
    run_id     = kwargs['run_id']
    data_root  = ti.xcom_pull(key='data_root', task_ids='load_data')
    xcom_key   = f'{model_name}_run{run_id}'
    model_path = ti.xcom_pull(key=f'{xcom_key}_model_path',
                               task_ids=f'train_{model_name}_run{run_id}')
    train_time = ti.xcom_pull(key=f'{xcom_key}_train_time',
                               task_ids=f'train_{model_name}_run{run_id}')

    transform   = transforms.Compose([transforms.ToTensor()])
    test_ds     = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MODEL_CLASSES[model_name]()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs   = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    eval_time = round(time.time() - t0, 3)

    accuracy = round(accuracy_score(all_labels, all_preds) * 100, 4)
    f1       = round(f1_score(all_labels, all_preds, average='macro') * 100, 4)

    print(f"[evaluate] model={model_name} run={run_id}")
    print(f"  Accuracy:   {accuracy:.4f}%")
    print(f"  F1-macro:   {f1:.4f}%")
    print(f"  Train time: {train_time}s")
    print(f"  Eval time:  {eval_time}s")
    print(f"  Total time: {round(train_time + eval_time, 3)}s")

    # Push metrics — dễ collect sau
    xk = f'{model_name}_run{run_id}'
    ti.xcom_push(key=f'{xk}_accuracy',  value=accuracy)
    ti.xcom_push(key=f'{xk}_f1',        value=f1)
    ti.xcom_push(key=f'{xk}_eval_time', value=eval_time)


# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id='mnist_classification_airflow',
    default_args=default_args,
    description='UC1 MNIST — 3 models × 3 runs, metrics: accuracy + F1 + timing',
    schedule_interval=None,
    catchup=False,
    tags=['mnist', 'pytorch', 'uc1'],
) as dag:

    t_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    for model_name in MODELS:
        for run_id in range(1, NUM_RUNS + 1):
            t_train = PythonOperator(
                task_id=f'train_{model_name}_run{run_id}',
                python_callable=train_model,
                op_kwargs={'model_name': model_name, 'run_id': run_id},
            )

            t_eval = PythonOperator(
                task_id=f'evaluate_{model_name}_run{run_id}',
                python_callable=evaluate_model,
                op_kwargs={'model_name': model_name, 'run_id': run_id},
            )

            t_load >> t_train >> t_eval
