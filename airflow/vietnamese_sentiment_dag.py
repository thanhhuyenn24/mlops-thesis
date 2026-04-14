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


def predict_sentiment(**kwargs):
    from transformers import pipeline
    import pandas as pd

    # Retrieve data path from XCom
    ti = kwargs['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')

    df = pd.read_csv(data_path)
    texts = df['sentence'].tolist()
    labels = df['sentiment'].tolist()

    # Load pre-trained PhoBERT-based Vietnamese sentiment model
    print("Loading PhoBERT model...")
    sentiment_pipeline = pipeline(
        "text-classification",
        model="wonrax/phobert-base-vietnamese-sentiment",
        tokenizer="wonrax/phobert-base-vietnamese-sentiment",
        device=-1,  # CPU inference
        truncation=True,
        max_length=256,
    )

    # Run inference on all samples
    print(f"Running inference on {len(texts)} samples...")
    preds = []
    scores = []
    for i, txt in enumerate(texts):
        result = sentiment_pipeline(txt[:512])[0]
        preds.append(result['label'])
        scores.append(result['score'])
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(texts)} samples")

    # Map predicted labels to integer IDs for accuracy computation
    label_map = {'NEG': 0, 'POS': 1, 'NEU': 2}
    pred_ids = [label_map.get(p, -1) for p in preds]
    correct = sum(p == l for p, l in zip(pred_ids, labels))
    accuracy = 100 * correct / len(labels)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average confidence score: {sum(scores)/len(scores):.4f}")

    # Pass accuracy to XCom for traceability
    ti.xcom_push(key='accuracy', value=accuracy)


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