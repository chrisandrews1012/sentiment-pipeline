# AWS NLP Sentiment Pipeline

![GitHub last commit](https://img.shields.io/github/last-commit/chrisandrews1012/sentiment-pipeline)
![GitHub repo size](https://img.shields.io/github/repo-size/chrisandrews1012/sentiment-pipeline)
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Lambda%20%7C%20SageMaker%20%7C%20Bedrock-orange)

An end-to-end NLP pipeline on AWS built to get hands-on experience with core services — not intended to be a novel contribution. It ingests IMDB movie reviews, runs sentiment analysis with DistilBERT, processes batches nightly, serves predictions via a live REST API, and uses Claude Sonnet 4.6 via Bedrock for richer structured output.

---

## Project Overview

The pipeline covers the full lifecycle of an ML workload on AWS: data ingestion into S3, model inference via SageMaker and a containerized Lambda function, a nightly EventBridge-scheduled batch job, real-time predictions through API Gateway, and a comparison between a fine-tuned HuggingFace model and a frontier LLM (Claude via Bedrock). Experiment runs are tracked in SageMaker Experiments and the Lambda deployment is fully automated through a GitHub Actions CI/CD workflow.

---

## Architecture

```
IMDB Dataset (HuggingFace)
         │
         ▼
      S3 (raw/)
         │
         ├──► SageMaker Notebook ──► DistilBERT inference + exploration
         │
         ├──► Lambda + EventBridge ──► nightly batch job (50 rows/run)
         │              │
         │              ▼
         │           S3 (results/)
         │
         ├──► API Gateway → Lambda ──► real-time REST API
         │
         └──► Bedrock / Claude Sonnet 4.6 ──► structured analysis → S3 (results/)
```

**AWS services used:** S3, SageMaker, Lambda, EventBridge, CloudWatch, API Gateway, Bedrock, ECR, IAM

---

## Installation and Setup

### Codes and Resources Used

- **Editor:** VS Code
- **Python Version:** 3.12
- **Package Manager:** [uv](https://github.com/astral-sh/uv)

### Python Packages Used

**General Purpose**
- `boto3` — AWS SDK for Python (S3, Lambda, Bedrock, SageMaker clients)

**Data Manipulation**
- `pandas`, `pyarrow` — dataframe operations and Parquet I/O
- `datasets` — HuggingFace Datasets library for loading IMDB

**Machine Learning**
- `transformers` — HuggingFace model and tokenizer loading (DistilBERT)
- `torch` — PyTorch inference backend
- `sagemaker` — SageMaker SDK for endpoint deployment and experiment tracking

### Local Setup

```bash
# Clone the repo
git clone https://github.com/chrisandrews1012/sentiment-pipeline.git
cd sentiment-pipeline

# Install dependencies with uv
uv sync

# Configure AWS credentials
aws configure
```

> **Note:** The SageMaker notebooks (`01_exploration.ipynb`, `02_deploy.ipynb`) are designed to run inside SageMaker Studio, not locally. The `src/` scripts and the Bedrock notebooks can be run locally with valid AWS credentials.

---

## Data

### Source Data

**IMDB Movie Reviews** — a benchmark dataset of 50,000 labeled movie reviews (positive/negative) published by Stanford. This project uses 500 rows sampled from the test split.

- Source: [HuggingFace `datasets` — `imdb`](https://huggingface.co/datasets/imdb)
- License: ACL 2011 paper (Andrew Maas et al.)
- Columns: `text` (review string), `label` (0 = negative, 1 = positive)

### Data Acquisition

Data is loaded directly from HuggingFace using the `datasets` library, shuffled with a fixed seed for reproducibility, and saved as Parquet.

```bash
# Download 500 IMDB reviews and upload to S3
uv run python src/ingest.py
```

### Data Preprocessing

The raw dataset arrives clean and labeled — no missing values or preprocessing required. Reviews are truncated to 512 tokens at inference time to fit the DistilBERT context window. The parquet format is used throughout for efficient columnar storage and fast reads in Lambda.

---

## Models

### DistilBERT

`distilbert-base-uncased-finetuned-sst-2-english` is a smaller, faster version of BERT fine-tuned on the SST-2 sentiment classification benchmark. It returns a binary label (`POSITIVE` / `NEGATIVE`) and a confidence score. Achieved **~89% accuracy** on 500 IMDB samples.

**Sample output:**
```json
{
  "text": "This movie was absolutely fantastic!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998
}
```

### Claude Sonnet 4.6 (via AWS Bedrock)

Claude goes beyond binary classification — it returns structured output with mixed sentiment support, key themes, a narrative summary, and a recommendation likelihood. Run on a sample of 20 reviews for a qualitative comparison.

**Sample output:**
```json
{
  "sentiment": "positive",
  "confidence": "high",
  "key_themes": ["acting", "emotional impact", "personal connection"],
  "summary": "A deeply moving film with outstanding performances.",
  "would_recommend": true
}
```

See [notebooks/04_model_comparison.ipynb](notebooks/04_model_comparison.ipynb) for a side-by-side comparison of both approaches.

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

## Code Structure

```
sentiment-pipeline/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── data/
│   └── imdb_reviews.parquet
├── lambda/
│   ├── handler.py
│   └── Dockerfile
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_deploy.ipynb
│   ├── 03_bedrock_analysis.ipynb
│   └── 04_model_comparison.ipynb
├── sagemaker/
│   ├── deploy_endpoint.py
│   └── experiment_tracking.py
├── src/
│   ├── setup_bucket.py
│   ├── ingest.py
│   └── bedrock_inference.py
├── .python-version
├── pyproject.toml
└── uv.lock
```

---

## Results and Evaluation

| Model | Accuracy |
|---|---|
| DistilBERT (Lambda) | ~89% on 500 samples |
| Claude Sonnet 4.6 (Bedrock) | Qualitative only |

DistilBERT is fast and cheap for high-volume binary classification. Claude adds qualitative depth — mixed sentiment, thematic extraction, and natural language summaries — at higher cost and latency. For production use, DistilBERT handles throughput while Claude can be reserved for edge cases or richer downstream analysis.

Full comparison: [notebooks/04_model_comparison.ipynb](notebooks/04_model_comparison.ipynb)

---

## Future Work

- Add a confidence threshold to route low-confidence DistilBERT predictions to Bedrock automatically
- Build a simple dashboard (Streamlit or Grafana) to visualize batch results from S3
- Extend to multi-class sentiment (1–5 star rating) using a fine-tuned model
- Add unit tests for the Lambda handler covering both invocation modes
- Evaluate cost/latency tradeoffs of SageMaker real-time endpoint vs. Lambda container

---

## Acknowledgments

- [HuggingFace `distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [HuggingFace IMDB dataset](https://huggingface.co/datasets/imdb) — Andrew Maas et al., ACL 2011
- [pragyy/datascience-readme-template](https://github.com/pragyy/datascience-readme-template) — README structure

---

## License

