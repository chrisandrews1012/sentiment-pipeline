import boto3
import pandas as pd
import json
from io import BytesIO

BUCKET = "candrews-sentiment-pipeline"

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
s3 = boto3.client("s3")

def analyze_review(text: str) -> dict:
    """
    Send a review to Claude via Bedrock and get structured analysis back.
    Claude can do much more than a binary classifier:
    - Mixed sentiment (not just positive/negative)
    - Specific themes mentioned
    - A human-readable summary
    - Recommendation likelihood
    """
    prompt = f"""Analyze the sentiment of this movie review.

    Respond with a raw JSON object only. Do not use markdown, do not use code fences, do not include any explanation. Start your response with {{ and end with }}.

    The JSON must have exactly these fields:
    - sentiment: one of "positive", "negative", or "mixed"
    - confidence: one of "high", "medium", or "low"
    - key_themes: list of up to 3 main topics mentioned (e.g. ["acting", "plot", "visuals"])
    - summary: one sentence summary of the review
    - would_recommend: true or false

    Review:
    {text}"""
    
    response = bedrock.invoke_model(
        modelId="global.anthropic.claude-sonnet-4-6",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    
    response_body = json.loads(response["body"].read())
    result_text = response_body["content"][0]["text"]
    
    # Parse the JSON Claude returned
    return json.loads(result_text)


def run_bedrock_pipeline():
    """
    
    """
    # Read raw data from S3
    obj = s3.get_object(Bucket=BUCKET, Key="raw/imdb_reviews.parquet")
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    
    # Run on a small sample first to control cost
    sample = df.head(20)
    
    results = []
    for i, row in sample.iterrows():
        print(f"Processing review {i+1}/20...")
        try:
            analysis = analyze_review(row["text"])
            analysis["original_label"] = row["label"]
            analysis["review_text"] = row["text"][:200]  # Store a snippet for reference
            results.append(analysis)
        except Exception as e:
            print(f"Error on review {i}: {e}")
            results.append({"error": str(e)})
            
    df_results = pd.DataFrame(results)
    print(df_results[["sentiment", "confidence", "key_themes", "would_recommend"]].head())
    
    # Write results back to S3
    buffer = BytesIO()
    df_results.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET, 
                  Key="results/bedrock_results.parquet",
                  Body=buffer.getvalue()
    )
    print("Bedrock results saved to S3")
    
if __name__ == "__main__":
    run_bedrock_pipeline()
    