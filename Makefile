.PHONY: help setup data predict

help:
	@echo "Available commands:"
	@echo "  make setup    Create S3 bucket and folder structure"
	@echo "  make data     Download IMDB reviews and upload to S3"
	@echo "  make predict  Run Bedrock inference on sample reviews"

setup:
	uv run python src/sentiment_pipeline/config.py

data:
	uv run python src/sentiment_pipeline/dataset.py

predict:
	uv run python src/sentiment_pipeline/modeling/predict.py
