import boto3

BUCKET = "candrews-sentiment-pipeline"    
s3 = boto3.client("s3")

s3.create_bucket(Bucket=BUCKET)
print(f"Created bucket: {BUCKET}")

# Create folder structure using empty placeholder objects
prefixes = ["raw/", "results/", "models/", "logs/"]
for prefix in prefixes:
    s3.put_object(Bucket=BUCKET, Key=prefix)
    print(f"Created folder: {prefix}")