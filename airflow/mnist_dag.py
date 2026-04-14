from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

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

    # DeepNN architecture as defined in the paper
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

    # Load and preprocess MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root='/tmp/mnist_data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Initialize model, loss function and optimizer
    model = DeepNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
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

    # Save trained model to temporary storage
    model_path = '/tmp/mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Pass model path to evaluate task via XCom
    ti = kwargs['ti']
    ti.xcom_push(key='model_path', value=model_path)


def evaluate_model(**kwargs):
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # DeepNN architecture must match training definition
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

    # Retrieve model path from XCom
    ti = kwargs['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')

    # Load and preprocess MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(
        root='/tmp/mnist_data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Load trained model weights
    model = DeepNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate model accuracy on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


with DAG(
    dag_id='mnist_classification_airflow',
    default_args=default_args,
    description='Train and evaluate MNIST with PyTorch (DeepNN)',
    schedule_interval=None,
    catchup=False,
    tags=['mnist', 'pytorch'],
) as dag:

    t1 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    t2 = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    t1 >> t2