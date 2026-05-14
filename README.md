# AWS NLP Sentiment Pipeline

I built this to get hands-on experience with AWS. The idea was to learn the core services (**S3**, **Lambda**, **SageMaker**, **EventBridge**, **API Gateway**, and **Bedrock**) by building an actual pipeline rather than just reading docs. It ingests IMDB movie reviews, runs sentiment analysis with DistilBERT, processes batches nightly, serves predictions via a live API, and uses Claude Sonnet 4.6 via Bedrock for richer structured output.

---

## Architecture

```
IMDB Dataset
     │
     ▼
  S3 (raw/)
     │
     ├──► SageMaker Notebook — DistilBERT inference
     │
     ├──► Lambda + EventBridge — nightly batch job
     │         │
     │         ▼
     │      S3 (results/)
     │
     ├──► API Gateway → Lambda — real-time REST API
     │
     └──► Bedrock / Claude Sonnet 4.6 — structured analysis
```

**AWS services used:** S3, SageMaker, Lambda, EventBridge, CloudWatch, API Gateway, Bedrock, ECR, IAM

---

## Models

### DistilBERT
`distilbert-base-uncased-finetuned-sst-2-english` is a smaller, faster version of BERT fine-tuned on sentiment classification. It returns a binary label (POSITIVE/NEGATIVE) and a confidence score. Achieved ~89% accuracy on 500 IMDB samples.

**Output:**
```json
{
  "text": "This movie was absolutely fantastic!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998
}
```

### Claude Sonnet 4.6 (via AWS Bedrock)
Claude goes beyond binary classification -- it returns structured output with mixed sentiment support, key themes, a summary, and a recommendation likelihood. Run on a sample of 20 reviews for comparison.

**Output:**
```json
{
  "sentiment": "positive",
  "confidence": "high",
  "key_themes": ["acting", "emotional impact", "personal connection"],
  "summary": "A deeply moving film with outstanding performances.",
  "would_recommend": true
}
```

See `notebooks/04_model_comparison.ipynb` for a side-by-side comparison of both approaches.

---

## Live API

The sentiment API is live and protected by an API key.

**Endpoint:** `POST https://vpjyzs30p0.execute-api.us-east-1.amazonaws.com/prod/predict`

```bash
curl -X POST https://vpjyzs30p0.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

---

## Project Structure

**Run locally:**
```
sentiment-pipeline/
├── .github/workflows/deploy.yml    ← CI/CD: auto-deploys Lambda on push
├── lambda/
│   ├── handler.py                  ← Lambda entry point (API + batch modes)
│   └── Dockerfile                  ← Docker image with DistilBERT baked in
├── notebooks/
│   ├── 03_bedrock_analysis.ipynb   ← Bedrock results analysis
│   └── 04_model_comparison.ipynb   ← DistilBERT vs Claude comparison
├── src/
│   ├── setup_bucket.py             ← Creates S3 bucket and folder structure
│   ├── ingest.py                   ← Downloads IMDB data and uploads to S3
│   └── bedrock_inference.py        ← Runs Claude via Bedrock, saves results to S3
```

**Run in SageMaker Studio:**
```
├── notebooks/
│   ├── 01_exploration.ipynb        ← DistilBERT inference
│   └── 02_deploy.ipynb             ← SageMaker endpoint deployment
├── sagemaker/
│   ├── deploy_endpoint.py          ← Deploys DistilBERT as SageMaker endpoint
│   └── experiment_tracking.py      ← Logs runs to SageMaker Experiments
```
