FROM public.ecr.aws/lambda/python:3.12

# install CPU-only torch first to keep image size down
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install boto3 pandas pyarrow transformers

# save model files directly to /opt/model at build time
# loading from a fixed path avoids HuggingFace cache mechanics entirely
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='distilbert-base-uncased-finetuned-sst-2-english'; AutoTokenizer.from_pretrained(m).save_pretrained('/opt/model'); AutoModelForSequenceClassification.from_pretrained(m).save_pretrained('/opt/model')"

COPY handler.py ${LAMBDA_TASK_ROOT}

CMD ["handler.handler"]