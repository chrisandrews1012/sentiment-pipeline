# This script must be run inside SageMaker Studio -- get_execution_role() is not available locally.

import sagemaker
from sagemaker.serve.model_builder import ModelBuilder

# SageMaker session handles authetication and region automatically
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

model_builder = ModelBuilder(
    model="distilbert-base-uncased-finetuned-sst-2-english",
    role_arn=role,
    sagemaker_session=sagemaker_session,
    instance_type="ml.m5.xlarge",
)

model = model_builder.build()

predictor = model.deploy(
    endpoint_name="sentiment-pipeline-endpoint",
)

print("Endpoint deployed: sentiment-pipeline-endpoint")

result = predictor.predict({"inputs": "This movie was absolutely fantastic!"})
print(f"Test prediction: {result}")

# Uncomment to delete the endpoint when done testing
# predictor.delete_endpoint()
