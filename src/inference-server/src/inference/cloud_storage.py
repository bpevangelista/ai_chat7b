from azure.storage.blob import BlobServiceClient

class AzureBlobServiceClient():
    def __init__(credential=None):
        self.account_url = "https://evangelista.blob.core.windows.net"
        self.service_client = BlobServiceClient(account_url, credential=credential or DefaultAzureCredential())

    def stream_from_blob(self, container_name, blob_name):
        blob_client = self.service_client.get_blob_client(container=container_name, blob=blob_name)
        stream = io.BytesIO()
        blob_client.download_blob().readinto(stream)
        return stream

    def blob_to_file(self, container_name, blob_name, file_name):
        blob_client = self.service_client.get_blob_client(container=container_name, blob=blob_name)
        with open(file_name, "wb") as file_data:
            blob_client.download_blob().readinto(file_data)