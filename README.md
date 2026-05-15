# AWS NLP Sentiment Pipeline

![GitHub last commit](https://img.shields.io/github/last-commit/chrisandrews1012/sentiment-pipeline)
![GitHub repo size](https://img.shields.io/github/repo-size/chrisandrews1012/sentiment-pipeline)
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Lambda%20%7C%20SageMaker%20%7C%20Bedrock-orange)

An end-to-end NLP pipeline on AWS built to get hands-on experience with core services — not intended to be a novel contribution. It ingests IMDB movie reviews, runs sentiment analysis with DistilBERT, processes batches nightly, serves predictions via a live REST API, and uses Claude Sonnet 4.6 via Bedrock for richer structured output.

## Models

Two approaches are compared: **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) for fast binary classification (~89% accuracy on 500 samples), and **Claude Sonnet 4.6 via Bedrock** for richer structured output with mixed sentiment support, key themes, and recommendation likelihood. See `notebooks/04_model_comparison.ipynb` for the side-by-side comparison.

## Setup

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

## Live API

**Endpoint:** `POST https://vpjyzs30p0.execute-api.us-east-1.amazonaws.com/prod/predict`

```bash
curl -X POST https://vpjyzs30p0.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

## Project Organization

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
