from azure.storage.blob import BlobServiceClient


storage_account_key = "1KkwHkZNcTsbjIUjHA1/JYL2PJ3GiM8RzahijVuHJ0+PH4q9UvfBt7FNso+pVaNl4nD/ARpP7V6h+AStY+D3Jw=="
storage_account_name = "jadeynovapi"
connection_string = "DefaultEndpointsProtocol=https;AccountName=jadeynovapi;AccountKey=1KkwHkZNcTsbjIUjHA1/JYL2PJ3GiM8RzahijVuHJ0+PH4q9UvfBt7FNso+pVaNl4nD/ARpP7V6h+AStY+D3Jw==;EndpointSuffix=core.windows.net"
container_name = "jadecontainerstorage"
# function to upload file to blob storage
def uploadToBlobStorage(file_path,file_name):
   blob_service_client = BlobServiceClient.from_connection_string(connection_string)
   blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
   with open(file_path,"rb") as data:
      blob_client.upload_blob(data)
      print(f"Uploaded {file_name}.")
      
      
uploadToBlobStorage('../TP2 - MLOps avec MLFlow/sample_data/california_housing_test.csv','california_test')
uploadToBlobStorage('../TP2 - MLOps avec MLFlow/sample_data/california_housing_train.csv','california_train')