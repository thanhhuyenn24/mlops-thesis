"""
UC1 MNIST — MLflow Pipeline
Chạy: python3 train_uc1.py
      python3 train_uc1.py --mode repeat
      python3 train_uc1.py --mode tc2
Kết quả log tự động lên MLflow server
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import argparse

from shared.models_mnist import SimpleNN, DeepNN, CNN, MODEL_CLASSES, HPARAMS

# CONFIG
CONFIG = {
    "tracking_uri":  "http://localhost:5000",
    "experiment":    "UC1_MNIST_MLflow",
    "data_path":     "./data",
    "num_runs":      3,

    "models": ["SimpleNN", "DeepNN", "CNN"],

    # TC2 — Config sweep
    "tc2_configs": [
        {"lr": 0.001, "batch_size": 32,  "epochs": 10},
        {"lr": 0.01,  "batch_size": 64,  "epochs": 10},
        {"lr": 0.05,  "batch_size": 128, "epochs": 10},
    ],
}


# FUNCTIONS

def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = torchvision.datasets.MNIST(
        CONFIG["data_path"], train=True,
        download=True, transform=transform)
    test = torchvision.datasets.MNIST(
        CONFIG["data_path"], train=False, transform=transform)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test,  batch_size=batch_size),
        len(test)
    )


def train_one_run(model_name, lr, batch_size, epochs, run_name):
    train_loader, test_loader, test_size = get_data(batch_size)
    model     = MODEL_CLASSES[model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=run_name):
        start_time = time.time()

        mlflow.log_param("model",      model_name)
        mlflow.log_param("lr",         lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs",     epochs)
        mlflow.log_param("optimizer",  "Adam")
        mlflow.log_param("platform",   "MLflow-GCP")

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        train_time = time.time() - start_time

        eval_start = time.time()
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                correct += (model(X).argmax(1) == y).sum().item()
        accuracy = correct / test_size
        eval_time = time.time() - eval_start

        pipeline_time = time.time() - start_time
        mlflow.log_metric("test_accuracy",          accuracy)
        mlflow.log_metric("train_time_seconds",     train_time)
        mlflow.log_metric("eval_time_seconds",      eval_time)
        mlflow.log_metric("pipeline_time_seconds",  pipeline_time)
        mlflow.pytorch.log_model(model, "model")

        print(f"     Accuracy={accuracy:.4f} | Time={pipeline_time:.1f}s")
        return accuracy, pipeline_time


# MAIN

def main(mode="all"):
    mlflow.set_tracking_uri(CONFIG["tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment"])

    if mode in ("all", "repeat"):
        print("\n PHẦN 1: Chạy lặp 9 runs (3 model × 3 lần) — đo reproducibility")
        for model_name in CONFIG["models"]:
            for i in range(1, CONFIG["num_runs"] + 1):
                print(f"\n  [{model_name}] Lần {i}/{CONFIG['num_runs']}")
                train_one_run(
                    model_name = model_name,
                    lr         = HPARAMS["lr"],
                    batch_size = HPARAMS["batch_size"],
                    epochs     = HPARAMS["epochs"],
                    run_name   = f"{model_name}_run{i}"
                )

    if mode in ("all", "tc2"):
        print("\n PHẦN 2: TC2 Config Sweep")
        for cfg in CONFIG["tc2_configs"]:
            run_name = f"TC2_lr{cfg['lr']}_batch{cfg['batch_size']}"
            print(f"\n  Config: {run_name}")
            train_one_run(
                model_name = "SimpleNN",
                run_name   = run_name,
                **cfg
            )

    print("\n Hoàn thành! Xem kết quả tại:", CONFIG["tracking_uri"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "repeat", "tc2"],
        default="all",
        help="all=chạy tất cả | repeat=chỉ lặp 9 runs | tc2=chỉ config sweep"
    )
    args = parser.parse_args()
    main(args.mode)
