import json
import boto3, sagemaker

client = boto3.client("sagemaker-runtime")

#content_type = "text/plain"
content_type = "application/json"
payload = {
    "inputs": "Amazon.com is the best",
}
response = client.invoke_endpoint(
    EndpointName="pygmalion-6b-aug29-endpoint", ContentType=content_type, Body=json.dumps(payload)
)
print(response["Body"].read())