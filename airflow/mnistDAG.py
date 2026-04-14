from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

def train_model(**kwargs):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # DeepNN model - đúng theo paper
    class DeepNN(nn.Module):
        def __init__(self):
            super(DeepNN, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root='/tmp/mnist_data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Train model
    model = DeepNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save model
    model_path = '/tmp/mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Pass model path sang evaluate_model qua XCom - đúng theo paper
    ti = kwargs['ti']
    ti.xcom_push(key='model_path', value=model_path)


def evaluate_model(**kwargs):
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    class DeepNN(nn.Module):
        def __init__(self):
            super(DeepNN, self).__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    ti = kwargs['ti']
    model