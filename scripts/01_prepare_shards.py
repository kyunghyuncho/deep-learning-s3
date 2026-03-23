import torchvision
import webdataset as wds
import os

def prepare_data():
    # Download CIFAR10
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    # We want to shard this out. Shards limit small file OS overhead.
    # webdataset.ShardWriter natively supports automatic chunking given a pattern.
    pattern = "cifar-train-%06d.tar"
    max_count = 1000
    
    print(f"Writing dataset to {pattern} with {max_count} items per shard...")
    
    with wds.ShardWriter(pattern, maxcount=max_count) as sink:
        for i, (img, label) in enumerate(dataset):
            # Webdataset encodes data effectively when following standard extensions.
            # "png" will convert the PIL image into a PNG bytes buffer.
            # "cls" will convert the label into string representation.
            sink.write({
                "__key__": f"{i:06d}",
                "png": img,
                "cls": label
            })
    
    print("Serialization complete.")
    print("\n--- Next Steps: Upload to S3 ---")
    print("Use the following Python code using boto3 inside your environment to upload:")
    print('''
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
''')

if __name__ == "__main__":
    prepare_data()
