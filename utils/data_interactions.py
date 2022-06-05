from google.cloud.storage import Client

def gcp_client():
    credentials = 'gcp_config/mlops-3-1ccb1337a897.json'
    return Client.from_service_account_json(json_credentials_path=credentials)


def upload_blob(storage_client: Client, bucket_name: str, source_file_name: str, destination_blob_name: str):
    """Uploads a file to the google storage bucket.""" 
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
 
    blob.upload_from_filename(source_file_name)
 
    print(f"{source_file_name} uploaded to Storage Bucket with Bob name {destination_blob_name} successfully.")


def download_blob(storage_client: Client, bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Downloads a blob."""
 
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
 
    print(f"{source_blob_name} downloaded to file path {destination_file_name} successfully")