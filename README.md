# AWS NLP Sentiment Pipeline

![GitHub last commit](https://img.shields.io/github/last-commit/chrisandrews1012/sentiment-pipeline)
![GitHub repo size](https://img.shields.io/github/repo-size/chrisandrews1012/sentiment-pipeline)
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Lambda%20%7C%20SageMaker%20%7C%20Bedrock-orange)

Built to get hands-on familiarity with core AWS services: S3, Lambda, SageMaker, EventBridge, API Gateway, CloudWatch, Bedrock, ECR, and IAM.

## Problem Statement

Sentiment analysis is a common NLP task in industry, used for product reviews, customer feedback, social media monitoring, and more. The challenge isn't just the model; it's operationalizing it: storing data reliably, automating batch jobs, serving predictions at low latency, and deciding when a simpler model is good enough versus when a more powerful one is worth the cost.

## Approach

Two models are compared:

- **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`): a lightweight fine-tuned BERT model for fast binary sentiment classification, deployed as a containerized Lambda function
- **Claude Sonnet 4.6 via Bedrock**: a frontier LLM used for richer structured output including mixed sentiment, key themes, a summary, and recommendation likelihood

The pipeline ingests 500 IMDB movie reviews into S3, runs nightly batch inference via Lambda and EventBridge, serves real-time predictions through API Gateway, and uses SageMaker for notebook-based exploration and experiment tracking. Deployment is automated via GitHub Actions CI/CD.

## Results

| Model | Accuracy |
|---|---|
| DistilBERT | ~89% on 500 samples |
| Claude Sonnet 4.6 | Qualitative only |

DistilBERT handles high-volume binary classification cheaply and quickly. Claude produces richer output but at higher cost and latency, making it better suited for edge cases or deeper analysis. See `notebooks/04_model_comparison.ipynb` for the full side-by-side comparison.

## How to Run

```bash
git clone https://github.com/chrisandrews1012/sentiment-pipeline.git
cd sentiment-pipeline
uv sync
aws configure
```

```bash
make setup    # Create S3 bucket and folder structure
make data     # Download IMDB reviews and upload to S3
```

> **Note:** The SageMaker notebooks (`01_exploration.ipynb`, `02_deploy.ipynb`) must be run inside SageMaker Studio.

**Live API**

```bash
curl -X POST https://vpjyzs30p0.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

## File Structure

```
sentiment-pipeline/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── data/
│   ├── raw/
│   │   └── imdb_reviews.parquet
│   ├── interim/
│   ├── processed/
│   └── external/
├── deploy/
│   ├── lambda/
│   │   ├── handler.py
│   │   └── Dockerfile
│   └── sagemaker/
│       ├── deploy_endpoint.py
│       └── experiment_tracking.py
├── models/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_deploy.ipynb
│   ├── 03_bedrock_analysis.ipynb
│   └── 04_model_comparison.ipynb
├── reports/
│   └── figures/
├── src/
│   └── sentiment_pipeline/
│       ├── config.py
│       ├── dataset.py
│       └── modeling/
│           └── predict.py
├── .python-version
├── Makefile
├── pyproject.toml
└── uv.lock
```
