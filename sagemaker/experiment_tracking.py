# This script must be run inside SageMaker Studio

import sagemaker
import boto3
from io import BytesIO
import pandas as pd
from transformers import pipeline
from sagemaker.experiments.run import Run

BUCKET = "candrews-sentiment-pipeline"

s3 = boto3.client("s3")
obj = s3.get_object(Bucket=BUCKET, Key="raw/imdb_reviews.parquet")
df = pd.read_parquet(BytesIO(obj["Body"].read()))

texts = df["text"].tolist()[:100]
labels = df["label"].tolist()[:100]

classifier = pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1)

with Run(
    experiment_name="sentiment-pipeline",
    run_name="distilbert-baseline",
    sagemaker_session=sagemaker.Session()
) as run:
    run.log_parameter("model", "distilbert-base-uncased-finetuned-sst-2-english")
    run.log_parameter("dataset", "imdb")
    run.log_parameter("sample_size", 100)
    
    results = classifier(texts, truncation=True, max_length=512)
    
    label_map = {"NEGATIVE": 0, "POSITIVE": 1}
    predictions = [label_map[r["label"]] for r in results]
    accuracy = sum(p == t for p, t in zip(predictions, labels)) / len(labels)
    
    run.log_metric("accuracy", accuracy)
    run.log_metric("avg_confidence", sum(r["score"] for r in results) / len(results))
    
    print(f"Logged run with accuracy: {accuracy:.4f}")
    