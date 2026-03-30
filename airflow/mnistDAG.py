from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

def train_model():
    print("Training MNIST model...")

with DAG(
    dag_id='mnist_classification_airflow',
    default_args=default_args,
    description='Train and evaluate MNIST with PyTorch',
    schedule_interval=None,
    catchup=False,
    tags=['mnist', 'pytorch'],
) as dag:
    t1 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )