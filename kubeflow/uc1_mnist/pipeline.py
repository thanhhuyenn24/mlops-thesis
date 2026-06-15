"""
Kubeflow Pipelines — UC1: MNIST Classification
12 runs: 9 baseline (3 models x3) + 3 TC2 sweep (SimpleNN).
"""

from kfp import dsl
from kfp.client import Client

MNIST_IMAGE = "localhost:32000/kfp-mnist:latest"


@dsl.component(base_image=MNIST_IMAGE)
def train_and_evaluate(
    model_name : str,
    lr         : float,
    batch_size : int,
    epochs     : int,
    run_tag    : str,
) -> str:
    import json, time, struct, gzip, urllib.request, os
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    # ── Model definitions ─────────────────────────────────────────────────────
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
            )
        def forward(self, x):
            return self.net(x.view(-1, 784))

    class DeepNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 10),
            )
        def forward(self, x):
            return self.net(x.view(-1, 784))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10)
            )
        def forward(self, x):
            return self.fc(self.conv(x.view(-1, 1, 28, 28)).view(-1, 64*7*7))

    MODEL_CLASSES = {"SimpleNN": SimpleNN, "DeepNN": DeepNN, "CNN": CNN}

    # ── Download MNIST manually ───────────────────────────────────────────────
    def load_mnist_images(path):
        with gzip.open(path, "rb") as f:
            f.read(4); n = struct.unpack(">I", f.read(4))[0]; f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 784).astype(np.float32) / 255.0

    def load_mnist_labels(path):
        with gzip.open(path, "rb") as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist"
    files = {
        "train-images": "train-images-idx3-ubyte.gz",
        "train-labels": "train-labels-idx1-ubyte.gz",
        "test-images":  "t10k-images-idx3-ubyte.gz",
        "test-labels":  "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs("/tmp/mnist", exist_ok=True)
    for key, fname in files.items():
        dest = f"/tmp/mnist/{fname}"
        if not os.path.exists(dest):
            urllib.request.urlretrieve(f"{base_url}/{fname}", dest)

    train_x = (load_mnist_images("/tmp/mnist/train-images-idx3-ubyte.gz") - 0.1307) / 0.3081
    train_y = load_mnist_labels("/tmp/mnist/train-labels-idx1-ubyte.gz")
    test_x  = (load_mnist_images("/tmp/mnist/t10k-images-idx3-ubyte.gz")  - 0.1307) / 0.3081
    test_y  = load_mnist_labels("/tmp/mnist/t10k-labels-idx1-ubyte.gz")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
        batch_size=256, shuffle=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    device    = torch.device("cpu")
    model     = MODEL_CLASSES[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    pipeline_start = time.time()
    train_start    = time.time()
    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
    train_time = time.time() - train_start

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_start = time.time()
    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    eval_time     = time.time() - eval_start
    pipeline_time = time.time() - pipeline_start

    result = {
        "run_id"          : run_tag,
        "model"           : model_name,
        "lr"              : lr,
        "batch_size"      : batch_size,
        "accuracy"        : round(correct / total, 6),
        "f1_macro"        : round(f1_score(all_labels, all_preds, average="macro"), 6),
        "train_time_s"    : round(train_time, 4),
        "eval_time_s"     : round(eval_time, 4),
        "pipeline_time_s" : round(pipeline_time, 4),
    }
    print("CSV_ROW:" + ",".join(str(result[k]) for k in [
        "run_id", "model", "lr", "batch_size",
        "accuracy", "f1_macro", "train_time_s", "eval_time_s", "pipeline_time_s",
    ]))
    print(json.dumps(result, indent=2))
    return json.dumps(result)


@dsl.pipeline(
    name        = "MNIST Classification — KFP UC1",
    description = "12 runs: 9 baseline (3 models x3) + 3 TC2 sweep (SimpleNN).",
)
def mnist_pipeline():
    HPARAMS     = {"lr": 0.001, "batch_size": 64, "epochs": 10}
    MODELS      = ["SimpleNN", "DeepNN", "CNN"]
    NUM_RUNS    = 3
    TC2_CONFIGS = [
        {"lr": 0.001, "batch_size": 32,  "epochs": 10},
        {"lr": 0.01,  "batch_size": 64,  "epochs": 10},
        {"lr": 0.05,  "batch_size": 128, "epochs": 10},
    ]

    prev_task = None

    for model_name in MODELS:
        for run_idx in range(1, NUM_RUNS + 1):
            run_tag = f"baseline_{model_name}_run{run_idx}"
            task = train_and_evaluate(
                model_name=model_name, lr=HPARAMS["lr"],
                batch_size=HPARAMS["batch_size"], epochs=HPARAMS["epochs"],
                run_tag=run_tag,
            )
            task.set_caching_options(enable_caching=False)
            task.set_display_name(run_tag)
            if prev_task is not None:
                task.after(prev_task)
            prev_task = task

    for sweep_idx, cfg in enumerate(TC2_CONFIGS, start=1):
        run_tag = f"sweep{sweep_idx}"
        task = train_and_evaluate(
            model_name="SimpleNN", lr=cfg["lr"],
            batch_size=cfg["batch_size"], epochs=cfg["epochs"],
            run_tag=run_tag,
        )
        task.set_caching_options(enable_caching=False)
        task.set_display_name(run_tag)
        task.after(prev_task)
        prev_task = task


if __name__ == "__main__":
    import kfp

    PIPELINE_YAML = "/home/ubuntu/mlops-thesis/kubeflow/mnist_pipeline.yaml"
    KFP_HOST      = "http://localhost:8888"

    kfp.compiler.Compiler().compile(mnist_pipeline, PIPELINE_YAML)
    print(f"[INFO] Compiled → {PIPELINE_YAML}")

    client = Client(host=KFP_HOST)
    run = client.create_run_from_pipeline_func(
        pipeline_func  = mnist_pipeline,
        run_name       = "UC1-MNIST-12runs",
        enable_caching = False,
    )
    print(f"[INFO] Submitted → run_id: {run.run_id}")
