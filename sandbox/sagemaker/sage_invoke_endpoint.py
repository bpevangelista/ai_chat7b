import json
import boto3, sagemaker

client = boto3.client("sagemaker-runtime")

content_type = "text/plain"
payload = "Amazon.com is the best"
response = client.invoke_endpoint(
    EndpointName="pygmalion-6b-aug28-endpoint", ContentType=content_type, Body=payload
)
print(response["Body"].read())