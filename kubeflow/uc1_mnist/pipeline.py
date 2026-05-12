"""
UC1 — MNIST Classification — Kubeflow Pipelines 2.4.0
Pipeline: load_data >> train_model >> evaluate_model
3 models (SimpleNN, DeepNN, CNN) × 3 runs = 9 runs
Metrics: accuracy, F1-macro, train_time, eval_time
"""

from kfp import dsl, compiler
from kfp.dsl import Output, Input, Metrics, Model, Dataset

# ── Packages cài trong mỗi container ────────────────────────────────────────
BASE_PACKAGES = [
    "numpy==1.26.4",
    "Pillow==10.3.0",
    "torch==2.2.2+cpu",
    "torchvision==0.17.2+cpu",
    "scikit-learn==1.4.2",
]
PIP_URLS = [
    "https://download.pytorch.org/whl/cpu",
    "https://pypi.org/simple",
]

# ── Hyperparameters — đồng nhất với Airflow/MLflow/Metaflow ─────────────────
LR         = 0.001
BATCH_SIZE = 64
EPOCHS     = 10


# ── Step 1: Load data ────────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=BASE_PACKAGES,
    pip_index_urls=PIP_URLS,
)
def load_data(output_dataset: Output[Dataset]):
    """Download MNIST và lưu train/test tensors."""
    import time
    import torch
    from torchvision import datasets, transforms

    t0 = time.time()
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds  = datasets.MNIST(root="/tmp/mnist", train=True,  download=True, transform=transform)
    test_ds   = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)

    # Lưu dưới dạng tensor để tránh re-download ở các step sau
    data = {
        "train_images": train_ds.data.float() / 255.0,
        "train_labels": train_ds.targets,
        "test_images":  test_ds.data.float()  / 255.0,
        "test_labels":  test_ds.targets,
    }
    torch.save(data, output_dataset.path)

    elapsed = round(time.time() - t0, 3)
    print(f"[load_data] Train: {len(train_ds)}, Test: {len(test_ds)}, time={elapsed}s")


# ── Step 2: Train model ──────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=BASE_PACKAGES,
    pip_index_urls=PIP_URLS,
)
def train_model(
    input_dataset: Input[Dataset],
    model_name: str,
    run_id: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    output_model:   Output[Model],
    output_metrics: Output[Metrics],
):
    """Fine-tune SimpleNN / DeepNN / CNN trên MNIST train set."""
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # ── Model definitions — giống models_mnist.py ───────────────────────────
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1  = nn.Linear(784, 128)
            self.fc2  = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = x.view(-1, 784)
            return self.fc2(self.relu(self.fc1(x)))

    class DeepNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1     = nn.Linear(784, 512)
            self.fc2     = nn.Linear(512, 256)
            self.fc3     = nn.Linear(256, 128)
            self.fc4     = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.3)
            self.relu    = nn.ReLU()
        def forward(self, x):
            x = x.view(-1, 784)
            x = self.relu(self.fc1(x)); x = self.dropout(x)
            x = self.relu(self.fc2(x)); x = self.dropout(x)
            x = self.relu(self.fc3(x))
            return self.fc4(x)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1   = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1     = nn.BatchNorm2d(32)
            self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2     = nn.BatchNorm2d(64)
            self.pool    = nn.MaxPool2d(2, 2)
            self.fc1     = nn.Linear(64 * 7 * 7, 128)
            self.fc2     = nn.Linear(128, 10)
            self.relu    = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 64 * 7 * 7)
            return self.fc2(self.dropout(self.relu(self.fc1(x))))

    MODEL_CLASSES = {"SimpleNN": SimpleNN, "DeepNN": DeepNN, "CNN": CNN}

    # ── Load data ────────────────────────────────────────────────────────────
    data         = torch.load(input_dataset.path)
    train_images = data["train_images"].unsqueeze(1)  # (N,1,28,28) cho CNN
    train_labels = data["train_labels"]

    train_ds     = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ── Train ────────────────────────────────────────────────────────────────
    model     = MODEL_CLASSES[model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"[train] model={model_name} run={run_id} epochs={epochs} lr={learning_rate} batch={batch_size}")

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    train_time = round(time.time() - t0, 3)
    print(f"[train] done. train_time={train_time}s")

    torch.save(model.state_dict(), output_model.path)
    output_metrics.log_metric("train_time_seconds", train_time)
    output_metrics.log_metric("final_loss",         round(avg_loss, 4))


# ── Step 3: Evaluate model ───────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=BASE_PACKAGES,
    pip_index_urls=PIP_URLS,
)
def evaluate_model(
    input_dataset: Input[Dataset],
    input_model:   Input[Model],
    model_name:    str,
    run_id:        int,
    output_metrics: Output[Metrics],
):
    """Evaluate model trên MNIST test set, log accuracy + F1-macro."""
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, f1_score

    # ── Model definitions — giống train step ────────────────────────────────
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1  = nn.Linear(784, 128)
            self.fc2  = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = x.view(-1, 784)
            return self.fc2(self.relu(self.fc1(x)))

    class DeepNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1     = nn.Linear(784, 512)
            self.fc2     = nn.Linear(512, 256)
            self.fc3     = nn.Linear(256, 128)
            self.fc4     = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.3)
            self.relu    = nn.ReLU()
        def forward(self, x):
            x = x.view(-1, 784)
            x = self.relu(self.fc1(x)); x = self.dropout(x)
            x = self.relu(self.fc2(x)); x = self.dropout(x)
            x = self.relu(self.fc3(x))
            return self.fc4(x)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1   = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1     = nn.BatchNorm2d(32)
            self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2     = nn.BatchNorm2d(64)
            self.pool    = nn.MaxPool2d(2, 2)
            self.fc1     = nn.Linear(64 * 7 * 7, 128)
            self.fc2     = nn.Linear(128, 10)
            self.relu    = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 64 * 7 * 7)
            return self.fc2(self.dropout(self.relu(self.fc1(x))))

    MODEL_CLASSES = {"SimpleNN": SimpleNN, "DeepNN": DeepNN, "CNN": CNN}

    # ── Load test data ───────────────────────────────────────────────────────
    data        = torch.load(input_dataset.path)
    test_images = data["test_images"].unsqueeze(1)
    test_labels = data["test_labels"]

    test_ds     = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model = MODEL_CLASSES[model_name]()
    model.load_state_dict(torch.load(input_model.path, map_location="cpu"))
    model.eval()

    all_preds, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    eval_time = round(time.time() - t0, 3)
    accuracy  = round(accuracy_score(all_labels, all_preds) * 100, 4)
    f1        = round(f1_score(all_labels, all_preds, average='macro') * 100, 4)

    print(f"[evaluate] model={model_name} run={run_id}")
    print(f"  Accuracy: {accuracy:.4f}%")
    print(f"  F1-macro: {f1:.4f}%")
    print(f"  Eval time: {eval_time}s")

    output_metrics.log_metric("accuracy",          accuracy)
    output_metrics.log_metric("f1_macro",          f1)
    output_metrics.log_metric("eval_time_seconds", eval_time)


# ── Pipeline definition ──────────────────────────────────────────────────────
BATCH_SIZE = 64  # dùng trong evaluate container

@dsl.pipeline(
    name="mnist-classification-kubeflow",
    description="UC1 MNIST — 3 models × 3 runs, metrics: accuracy + F1 + timing",
)
def mnist_pipeline(
    epochs:        int   = EPOCHS,
    learning_rate: float = LR,
    batch_size:    int   = BATCH_SIZE,
):
    # Step 1: load data một lần, dùng chung cho tất cả runs
    load_task = load_data()
    load_task.set_memory_limit("1Gi")
    load_task.set_cpu_limit("1")

    for model_name in ["SimpleNN", "DeepNN", "CNN"]:
        for run_id in range(1, 4):
            train_task = train_model(
                input_dataset=load_task.outputs["output_dataset"],
                model_name=model_name,
                run_id=run_id,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            train_task.set_memory_limit("2Gi")
            train_task.set_cpu_limit("2")
            train_task.after(load_task)

            eval_task = evaluate_model(
                input_dataset=load_task.outputs["output_dataset"],
                input_model=train_task.outputs["output_model"],
                model_name=model_name,
                run_id=run_id,
            )
            eval_task.set_memory_limit("1Gi")
            eval_task.set_cpu_limit("1")
            eval_task.after(train_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path="mnist_pipeline.yaml",
    )
    print("Compiled: mnist_pipeline.yaml")