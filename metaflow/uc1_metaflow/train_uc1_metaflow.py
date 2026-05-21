"""
UC1 MNIST — Metaflow Pipeline (Local Mode trên GCP VM)

Chạy:
  python3 train_uc1_metaflow.py run
  python3 train_uc1_metaflow.py run --model DeepNN
  python3 train_uc1_metaflow.py run --model CNN
  python3 train_uc1_metaflow.py run --lr 0.01 --batch_size 128
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

from metaflow import FlowSpec, step, Parameter
import time


class MNISTFlow(FlowSpec):
    """Pipeline UC1 MNIST — Metaflow"""

    model_name = Parameter('model', default='SimpleNN',
                           help='Model: SimpleNN, DeepNN, or CNN')
    lr = Parameter('lr', default=0.001, type=float,
                   help='Learning rate')
    batch_size = Parameter('batch_size', default=64, type=int,
                           help='Batch size')
    epochs = Parameter('epochs', default=10, type=int,
                       help='Number of epochs')

    @step
    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*50}")
        print(f"  UC1 MNIST — Metaflow (GCP VM)")
        print(f"  Model: {self.model_name}")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, epochs={self.epochs}")
        print(f"  Optimizer: Adam")
        print(f"{'='*50}")
        self.next(self.train_and_evaluate)

    @step
    def train_and_evaluate(self):
        """Load data + Train + Evaluate (gộp 1 step tránh pickle error)"""
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as transforms
        from shared.models_mnist import MODEL_CLASSES

        # --- Load Data ---
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            './data', train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size)

        print(f"  Data: {len(train_dataset)} train, {len(test_dataset)} test")

        # --- Train ---
        model = MODEL_CLASSES[self.model_name]()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print(f"  Training {self.model_name}...")
        train_start = time.time()
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}")

        self.train_time = time.time() - train_start

        # --- Evaluate ---
        eval_start = time.time()
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                correct += (model(X).argmax(1) == y).sum().item()

        self.accuracy = correct / len(test_dataset)
        self.eval_time = time.time() - eval_start
        self.pipeline_time = time.time() - self.start_time

        print(f"\n  Accuracy: {self.accuracy:.4f}")
        print(f"  Train time: {self.train_time:.1f}s")
        print(f"  Eval time: {self.eval_time:.1f}s")
        print(f"  Pipeline time: {self.pipeline_time:.1f}s")
        self.next(self.end)

    @step
    def end(self):
        print(f"\n{'='*50}")
        print(f"  FINAL RESULT")
        print(f"  Model: {self.model_name}")
        print(f"  Accuracy: {self.accuracy:.4f}")
        print(f"  Train time: {self.train_time:.1f}s")
        print(f"  Eval time: {self.eval_time:.1f}s")
        print(f"  Pipeline time: {self.pipeline_time:.1f}s")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, epochs={self.epochs}, optimizer=Adam")
        print(f"  Platform: Metaflow Local Mode (GCP VM)")
        print(f"{'='*50}")


if __name__ == '__main__':
    MNISTFlow()
