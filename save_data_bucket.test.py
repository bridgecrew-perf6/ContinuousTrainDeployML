from google.cloud.storage import Client



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

if __name__ == "__main__":

    bucket = 'project-capstone-fbf'
    source_file_name = 'synthetic_signal.py'
    destination= 'models/' + source_file_name
    source_file_path = 'src/' + source_file_name
    credentials = 'src/gcp_config/mlops-3-1ccb1337a897.json'
    client = Client.from_service_account_json(json_credentials_path=credentials)

    # upload_blob(client, bucket, source_file_path, destination)

    gcp_origin = 'models/papers cariad'
    local_destination = 'deprecated/papers_cariad.txt'

    download_blob(client, bucket, destination, local_destination)







# bucket = client.get_bucket(bucket)
# object_name_in_gcs_bucket = bucket.blob(destination+source_file)
# object_name_in_gcs_bucket.upload_from_filename(source_file)

