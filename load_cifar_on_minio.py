import boto3
import glob

s3 = boto3.client(
    's3', 
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

# NOTE: create the bucket "cifar-streaming" in the MinIO UI first, or use s3.create_bucket(Bucket="cifar-streaming")
try:
    s3.create_bucket(Bucket="cifar-streaming")
except:
    pass

for file in glob.glob("cifar-train-*.tar"):
    print(f"Uploading {file}...")
    s3.upload_file(file, "cifar-streaming", file)
print("Uploads complete!")