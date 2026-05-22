"""
UC1 — MNIST Classification — Apache Airflow
============================================
Models : SimpleNN, DeepNN, CNN (defined in shared/models_mnist.py)
Dataset: torchvision MNIST (standard train/test split)
Runs   : 3 models × 3 repeats = 9 baseline runs (TC7)
         + 3 hyperparameter sweep configs on SimpleNN (TC2)
         = 12 runs total

Pipeline per run: load_data >> train_<model>_<run> >> evaluate_<model>_<run>
"""

import sys
sys.path.insert(0, '/home/airflow/airflowServer/dags/shared')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_PATH = '/home/airflow/airflowServer/dags/shared'
DATA_ROOT   = '/tmp/mnist_data'

default_args = {
    'owner'     : 'airflow',
    'start_date': datetime(2024, 1, 1),
}


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def load_data(**kwargs):
    """
    Download MNIST via torchvision (cached after first run).
    Pushes the data root path and load time to XCom.
    """
    import sys; sys.path.insert(0, SHARED_PATH)
    import time
    from torchvision import datasets, transforms

    t0        = time.time()
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    elapsed = round(time.time() - t0, 3)
    print(f"[load_data] train={len(train_ds)} test={len(test_ds)} time={elapsed}s")

    kwargs['ti'].xcom_push(key='data_root',   value=DATA_ROOT)
    kwargs['ti'].xcom_push(key='load_time_s', value=elapsed)


def train_model(model_name: str, run_id: int, hparams: dict, **kwargs):
    """
    Train the specified model for one run using the provided hyperparameters.
    Saves model weights to /tmp and pushes timing metadata to XCom.
    """
    import sys; sys.path.insert(0, SHARED_PATH)
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from models_mnist import MODEL_CLASSES

    ti        = kwargs['ti']
    data_root = ti.xcom_pull(key='data_root', task_ids='load_data')

    LR         = hparams['lr']
    BATCH_SIZE = hparams['batch_size']
    EPOCHS     = hparams['epochs']

    transform    = transforms.Compose([transforms.ToTensor()])
    train_ds     = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = MODEL_CLASSES[model_name]()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"[train | {model_name} run={run_id}] lr={LR} batch={BATCH_SIZE} epochs={EPOCHS}")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{EPOCHS} — loss={total_loss / len(train_loader):.4f}")

    train_time = round(time.time() - t0, 3)
    print(f"[train | {model_name} run={run_id}] done. train_time={train_time}s")

    model_path = f'/tmp/mnist_{model_name}_run{run_id}.pth'
    torch.save(model.state_dict(), model_path)

    xk = f'{model_name}_run{run_id}'
    ti.xcom_push(key=f'{xk}_model_path', value=model_path)
    ti.xcom_push(key=f'{xk}_train_time', value=train_time)
    ti.xcom_push(key=f'{xk}_lr',         value=LR)
    ti.xcom_push(key=f'{xk}_batch_size', value=BATCH_SIZE)


def evaluate_model(model_name: str, run_id: int, hparams: dict, **kwargs):
    """
    Evaluate a trained model on the MNIST test set.
    Logs accuracy, F1-macro, and pipeline timing required by TC7.
    """
    import sys; sys.path.insert(0, SHARED_PATH)
    import time
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, f1_score
    from models_mnist import MODEL_CLASSES

    ti        = kwargs['ti']
    data_root = ti.xcom_pull(key='data_root',    task_ids='load_data')
    load_time = ti.xcom_pull(key='load_time_s',  task_ids='load_data')

    xk         = f'{model_name}_run{run_id}'
    train_id   = f'train_{model_name}_run{run_id}'
    model_path = ti.xcom_pull(key=f'{xk}_model_path', task_ids=train_id)
    train_time = ti.xcom_pull(key=f'{xk}_train_time', task_ids=train_id)

    BATCH_SIZE = hparams['batch_size']

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
            _, predicted = torch.max(model(images), 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    eval_time     = round(time.time() - t0, 3)
    accuracy      = round(accuracy_score(all_labels, all_preds) * 100, 4)
    f1            = round(f1_score(all_labels, all_preds, average='macro') * 100, 4)
    pipeline_time = round(load_time + train_time + eval_time, 3)

    print(f"[evaluate | {model_name} run={run_id}]")
    print(f"  model         : {model_name}")
    print(f"  lr            : {hparams['lr']}")
    print(f"  batch_size    : {hparams['batch_size']}")
    print(f"  Accuracy      : {accuracy:.4f}%")
    print(f"  F1-macro      : {f1:.4f}%")
    print(f"  Load time     : {load_time}s")
    print(f"  Train time    : {train_time}s")
    print(f"  Eval time     : {eval_time}s")
    print(f"  Pipeline time : {pipeline_time}s")

    ti.xcom_push(key=f'{xk}_accuracy',      value=accuracy)
    ti.xcom_push(key=f'{xk}_f1',            value=f1)
    ti.xcom_push(key=f'{xk}_eval_time',     value=eval_time)
    ti.xcom_push(key=f'{xk}_pipeline_time', value=pipeline_time)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

# Baseline runs: 3 models × 3 repeats with default hyperparameters (TC7)
# Sweep runs   : SimpleNN with 3 lr/batch configs (TC2)

from models_mnist import MODELS, NUM_RUNS, HPARAMS as DEFAULT_HPARAMS, TC2_CONFIGS

BASELINE_RUNS = [
    (model, run_id, DEFAULT_HPARAMS)
    for model  in MODELS
    for run_id in range(1, NUM_RUNS + 1)
]

SWEEP_RUNS = [
    ('SimpleNN', f'sweep{i + 1}', TC2_CONFIGS[i])
    for i in range(len(TC2_CONFIGS))
]

ALL_RUNS = BASELINE_RUNS + SWEEP_RUNS   # 9 baseline + 3 sweep = 12 total

with DAG(
    dag_id          ='mnist_classification_airflow',
    default_args    =default_args,
    description     ='UC1 MNIST — 9 baseline runs (TC7) + 3 sweep runs (TC2) = 12 total',
    schedule_interval=None,
    catchup         =False,
    tags            =['mnist', 'pytorch', 'uc1', 'tc2', 'tc7'],
) as dag:

    t_load = PythonOperator(
        task_id        ='load_data',
        python_callable=load_data,
    )

    for model_name, run_id, hparams in ALL_RUNS:
        t_train = PythonOperator(
            task_id         =f'train_{model_name}_run{run_id}',
            python_callable =train_model,
            op_kwargs       ={
                'model_name': model_name,
                'run_id'    : run_id,
                'hparams'   : hparams,
            },
        )
        t_eval = PythonOperator(
            task_id         =f'evaluate_{model_name}_run{run_id}',
            python_callable =evaluate_model,
            op_kwargs       ={
                'model_name': model_name,
                'run_id'    : run_id,
                'hparams'   : hparams,
            },
        )

        t_load >> t_train >> t_eval