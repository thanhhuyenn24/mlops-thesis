from kfp import dsl, compiler
from kfp.dsl import Output, Input, Metrics, Model


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "numpy==1.26.4",
        "Pillow==10.3.0",
        "torch==2.2.2+cpu",
        "torchvision==0.17.2+cpu",
    ],
    pip_index_urls=[
        "https://download.pytorch.org/whl/cpu",
        "https://pypi.org/simple",
    ],
)
def train_model(
    epochs: int,
    learning_rate: float,
    batch_size: int,
    output_model: Output[Model],
    output_metrics: Output[Metrics],
):
    """Train DeepNN on MNIST. Mirrors Airflow train_model task."""
    import time
    import numpy as np  # noqa
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print(f"numpy version: {np.__version__}")
    print(f"torch version: {torch.__version__}")

    class DeepNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),  nn.ReLU(),
                nn.Linear(64, 10),
            )
        def forward(self, x):
            return self.net(x)

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="/tmp/mnist", train=True,
                               download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0)

    model = DeepNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    avg_loss = 0.0
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    train_time = time.time() - start
    torch.save(model.state_dict(), output_model.path)
    output_metrics.log_metric("train_time_seconds", round(train_time, 2))
    output_metrics.log_metric("final_loss", round(avg_loss, 4))
    print(f"Training complete in {train_time:.2f}s")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "numpy==1.26.4",
        "Pillow==10.3.0",
        "torch==2.2.2+cpu",
        "torchvision==0.17.2+cpu",
    ],
    pip_index_urls=[
        "https://download.pytorch.org/whl/cpu",
        "https://pypi.org/simple",
    ],
)
def evaluate_model(
    input_model: Input[Model],
    output_metrics: Output[Metrics],
):
    """Evaluate DeepNN on MNIST test set. Mirrors Airflow evaluate_model task."""
    import time
    import numpy as np  # noqa
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print(f"numpy version: {np.__version__}")

    class DeepNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),  nn.ReLU(),
                nn.Linear(64, 10),
            )
        def forward(self, x):
            return self.net(x)

    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root="/tmp/mnist", train=False,
                              download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256,
                              shuffle=False, num_workers=0)

    model = DeepNN()
    model.load_state_dict(torch.load(input_model.path, map_location="cpu"))
    model.eval()

    start = time.time()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    eval_time = time.time() - start
    output_metrics.log_metric("accuracy", round(accuracy, 4))
    output_metrics.log_metric("eval_time_seconds", round(eval_time, 2))
    print(f"Accuracy: {accuracy:.4f} | Eval time: {eval_time:.2f}s")


@dsl.pipeline(
    name="mnist-classification-pipeline",
    description="UC1: DeepNN MNIST - KFP 2.4.0",
)
def mnist_pipeline(
    epochs: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 64,
):
    train_task = train_model(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    train_task.set_memory_limit("2Gi")
    train_task.set_cpu_limit("2")

    eval_task = evaluate_model(
        input_model=train_task.outputs["output_model"],
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
