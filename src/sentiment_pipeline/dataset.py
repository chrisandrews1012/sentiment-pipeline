import boto3
import pandas as pd
from datasets import load_dataset

BUCKET = "candrews-sentiment-pipeline"

# Load 500 rows from the IMBD test split
# Using test split so the data is clean and labeled 
df = load_dataset("imdb", split="test").shuffle(seed=42).select(range(500)).to_pandas()

print(f"Loaded {len(df)} reviews")
print(df.head())

# Save locally first to verify it looks right 
df.to_parquet("data/raw/imdb_reviews.parquet", index=False)
print("Saved locally to data/raw/imdb_reviews.parquet")

# Upload to S3
# Use upload_file for local files
s3 = boto3.client("s3")
s3.upload_file("data/raw/imdb_reviews.parquet",
               BUCKET, "raw/imdb_reviews.parquet")

print(f"Uploaded to s3://{BUCKET}/raw/imdb_reviews.parquet")
