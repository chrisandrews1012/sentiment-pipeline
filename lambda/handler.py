import boto3
import pandas as pd
import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from transformers import pipeline

BUCKET = "candrews-sentiment-pipeline"
BATCH_SIZE = 50
CURSOR_KEY = "logs/cursor.json"

classifer = pipeline("sentiment-analysis",
                     model="/opt/model",
                     tokenizer="/opt/model",
                     device=-1)

def read_cursor(s3: Any) -> int:
    """
    Read the current cursor position from S3.

    :param s3: boto3 S3 client
    :type s3: Any
    :returns: index of the next row to process, or 0 if no cursor exists yet
    :rtype: int
    """
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=CURSOR_KEY)
        return json.loads(obj["Body"].read())["last_index"]
    except s3.exceptions.NoSuchKey:
        return 0

def write_cursor(s3: Any, index: int) -> None:
    """
    Persist the cursor position to S3.

    :param s3: boto3 S3 client
    :type s3: Any
    :param index: index of the next row to process on the following run
    :type index: int
    :returns: None
    :rtype: None
    """
    s3.put_object(
        Bucket=BUCKET,
        Key=CURSOR_KEY,
        Body=json.dumps({"last_index": index})
    )

def handler(event: dict, context: Any) -> dict:
    """
    Lambda entry point. Handles two invocation modes:
    - API Gateway: event has a body with a text field -- returns a single prediction
    - EventBridge: event is empty -- runs the nightly batch job

    :param event: event data passed by the invoker (e.g. API Gateway, EventBridge)
    :type event: dict
    :param context: Lambda runtime information (remaining time, memory, etc.)
    :type context: Any
    :returns: response dict with statusCode and body
    :rtype: dict
    """
    s3 = boto3.client("s3")
    
    # API Gateway path -- single real-time prediction
    if event.get("body"):
        body = json.loads(event["body"])
        text = body.get("text", "")
        
        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No text provided"})
            }
        
        result = classifer(text, truncation=True, max_length=512)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "text": text,
                "sentiment": result[0]["label"],
                "confidence": round(result[0]["score"], 4)
            })
        }
        
    # EventBridge path -- batch processing
    obj = s3.get_object(Bucket=BUCKET, Key="raw/imdb_reviews.parquet")
    df_full = pd.read_parquet(BytesIO(obj["Body"].read()))

    start = read_cursor(s3)
    if start >= len(df_full):
        start = 0

    end = min(start + BATCH_SIZE, len(df_full))
    batch = df_full.iloc[start:end].copy()
    print(f"Processing rows {start}-{end-1} of {len(df_full)}")

    results = classifer(batch["text"].tolist(), truncation=True, max_length=512)
    batch["predicted_label"] = [r["label"] for r in results]
    batch["confidence"] = [r["score"] for r in results]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_key = f"results/imdb_results_{timestamp}.parquet"
    buffer = BytesIO()
    batch.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key=result_key, Body=buffer.getvalue())

    write_cursor(s3, end % len(df_full))
    print(f"Results written to {result_key}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Inference complete",
            "rows_processed": len(batch),
            "next_index": end % len(df_full)
        })
    }
