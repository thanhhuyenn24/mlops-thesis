from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}


def load_data(**kwargs):
    from datasets import load_dataset
    import pandas as pd
    import os

    print("INFO: Initiating extraction of the UIT-VSFC dataset.")

    try:
        # Fetch the training split of the Vietnamese Students' Feedback Corpus (UIT-VSFC)
        ds = load_dataset("uitnlp/vietnamese_students_feedback", split="train")
        df = ds.to_pandas()

        print(f"INFO: Successfully retrieved {len(df)} records from the source dataset.")

        # Ensure all strictly required columns exist prior to downstream processing
        required_cols = {"sentence", "sentiment"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

        # Execute stratified sampling to maintain a balanced class distribution
        sample_df = (
            df.groupby("sentiment", group_keys=False)
              .apply(lambda x: x.sample(min(len(x), 167), random_state=42))
              .reset_index(drop=True)
        )

        print(f"INFO: Dataset successfully downsampled to {len(sample_df)} records.")

        # Persist the sampled dataframe to local storage for inter-task communication
        output_path = "/tmp/vsfc_sample.csv"
        sample_df.to_csv(output_path, index=False)

        # Register the file path in Airflow XCom for the subsequent inference task
        ti = kwargs["ti"]
        ti.xcom_push(key="data_path", value=output_path)

        print(f"INFO: Sampled dataset successfully persisted to {output_path}")

    except Exception as e:
        print(f"CRITICAL: Data loading pipeline encountered an error: {e}")
        raise

def predict_sentiment(**kwargs):
    from transformers import pipeline
    import pandas as pd

    # Retrieve the intermediate dataset filepath registered by the upstream task
    ti = kwargs['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')

    df = pd.read_csv(data_path)
    texts = df['sentence'].tolist()
    labels = df['sentiment'].tolist()

    # Initialize the PhoBERT-based text-classification pipeline for Vietnamese sentiment analysis
    print("INFO: Initializing the Hugging Face PhoBERT sentiment analysis pipeline...")
    sentiment_pipeline = pipeline(
        "text-classification",
        model="wonrax/phobert-base-vietnamese-sentiment",
        tokenizer="wonrax/phobert-base-vietnamese-sentiment",
        device=-1,  # CPU inference
        truncation=True,
        max_length=256,
    )

    # Execute batch inference across the retrieved dataset
    print(f"INFO: Commencing model inference on {len(texts)} text samples.")
    preds = []
    scores = []
    for i, txt in enumerate(texts):
        result = sentiment_pipeline(txt[:512])[0]
        preds.append(result['label'])
        scores.append(result['score'])
        if (i + 1) % 50 == 0:
            print(f"INFO: Inference progress: Processed {i+1}/{len(texts)} samples.")

    # Translate the predicted text labels into integer IDs to compute the evaluation metric
    label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
    pred_ids = [label_map.get(p, -1) for p in preds]
    correct = sum(p == l for p, l in zip(pred_ids, labels))
    accuracy = 100 * correct / len(labels)

    print(f"INFO: Evaluation complete. Overall Model Accuracy: {accuracy:.2f}%")
    print(f"INFO: Evaluation complete. Average Model Confidence Score: {sum(scores)/len(scores):.4f}")

    # Log the calculated evaluation metrics to XCom for pipeline observability
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