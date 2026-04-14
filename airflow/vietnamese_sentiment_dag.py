from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}


def load_data(**kwargs):
    import pandas as pd
    import requests
    import os

    url = "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/main/data/train-00000-of-00001.parquet"
    local_path = "/tmp/vsfc_data.parquet"

    print(f"Downloading dataset from {url}...")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print("Download finished.")
    else:
        raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

    df = pd.read_parquet(local_path)

    sample_df = df.groupby('sentiment').apply(
        lambda x: x.sample(min(len(x), 167), random_state=42)
    ).reset_index(drop=True).head(500)

    output_path = '/tmp/vsfc_sample.csv'
    sample_df.to_csv(output_path, index=False)
    
    ti = kwargs['ti']
    ti.xcom_push(key='data_path', value=output_path)
    print(f"Successfully prepared {len(sample_df)} samples.")


with DAG(
    dag_id='vietnamese_sentiment_airflow',
    default_args=default_args,
    description='Vietnamese Sentiment Analysis with PhoBERT (UIT-VSFC)',
    schedule=None,
    catchup=False,
    tags=['phobert', 'sentiment', 'vietnamese'],
) as dag:

    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    t2 = PythonOperator(
        task_id='predict_sentiment',
        python_callable=predict_sentiment,
    )

    t1 >> t2