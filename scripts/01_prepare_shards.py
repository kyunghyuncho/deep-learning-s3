"""
01_prepare_shards.py — CIFAR-10 WebDataset Shard Generator
===========================================================

PURPOSE:
    This script downloads the CIFAR-10 dataset and repackages it into
    WebDataset-compatible .tar shard files, solving the "small files problem"
    that plagues deep learning data pipelines.

BACKGROUND — THE SMALL FILES PROBLEM:
    CIFAR-10 contains 60,000 images (50,000 train + 10,000 test). If stored
    as individual files, loading them over a network incurs:

        T_total = N × (T_seek + S_file / B_network)

    where T_seek is the per-file latency (HTTP roundtrip, DNS lookup, TCP
    handshake, S3 API call). With N=60,000 and T_seek ≈ 50ms, just the
    seek overhead alone is 3,000 seconds (50 minutes!) of GPU idle time.

    By packing 1,000 images into each .tar archive (shard), we reduce
    the number of network requests from 60,000 to just 50:

        T_total ≈ T_seek_archive + (N × S_file) / B_network

    The seek penalty is amortized across the entire shard, and the data
    transfer becomes a single sequential read — exactly what HDDs, SSDs,
    and network streams are optimized for.

WHAT IS WEBDATASET?
    WebDataset (https://github.com/webdataset/webdataset) is a PyTorch-
    compatible library that stores training samples inside standard POSIX
    .tar archives. Each sample is a group of files sharing the same key:

        000042.png    (the image)
        000042.cls    (the class label)

    The .tar format was chosen because:
    1. It is a sequential format — files are concatenated end-to-end with
       minimal headers, enabling efficient streaming reads.
    2. It requires no index — unlike ZIP, you don't need to seek to the
       end of the file to find the directory. This makes it ideal for
       streaming over HTTP (you start reading from byte 0).
    3. It is universally supported — every OS and programming language
       can read .tar files.

USAGE:
    python scripts/01_prepare_shards.py
"""

import torchvision    # PyTorch's computer vision library (datasets, transforms)
import webdataset as wds  # WebDataset library for .tar shard I/O
import os             # File system utilities


def prepare_data():
    """
    Downloads CIFAR-10 and writes it as WebDataset shards.

    CIFAR-10 DATASET:
        - 10 classes: airplane, automobile, bird, cat, deer, dog, frog,
          horse, ship, truck
        - 50,000 training images, 10,000 test images
        - Each image is 32×32 pixels, 3 color channels (RGB)
        - Originally collected by Alex Krizhevsky, Vinod Nair, and
          Geoffrey Hinton (University of Toronto, 2009)

    SHARD CONFIGURATION:
        - Pattern: cifar-train-%06d.tar (e.g., cifar-train-000000.tar)
        - Max items per shard: 1,000
        - Total shards: 50 (50,000 images / 1,000 per shard)
        - Each shard is roughly 3 MB (1,000 × ~3KB per PNG)
    """
    # torchvision.datasets.CIFAR10 automatically:
    # 1. Downloads the dataset from https://www.cs.toronto.edu/~kriz/cifar.html
    # 2. Extracts the binary pickle files
    # 3. Returns an iterable of (PIL.Image, int) tuples
    # The `root` parameter specifies where to cache the download
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    # ShardWriter pattern: %06d means zero-padded 6-digit shard number
    # This produces: cifar-train-000000.tar, cifar-train-000001.tar, ...
    pattern = "cifar-train-%06d.tar"
    
    # maxcount: number of samples per shard before rolling to the next file
    # This is the key parameter controlling the seek/bandwidth tradeoff:
    # - Too small (e.g., 10): too many shards, seek penalty returns
    # - Too large (e.g., 100,000): single shard too big for parallel loading
    # - Sweet spot: 1,000-10,000 samples per shard (empirical best practice)
    max_count = 1000
    
    print(f"Writing dataset to {pattern} with {max_count} items per shard...")
    
    # ShardWriter is a context manager that handles:
    # 1. Opening a new .tar file when the current one reaches maxcount
    # 2. Serializing Python objects to the appropriate file format
    # 3. Writing the tar header + data for each sample
    with wds.ShardWriter(pattern, maxcount=max_count) as sink:
        for i, (img, label) in enumerate(dataset):
            # Each sample is a dictionary with special keys:
            #
            # "__key__": A unique identifier for this sample within the shard.
            #            WebDataset uses this to group related files together.
            #            Format: "000042" groups 000042.png and 000042.cls
            #
            # "png":     The file extension determines the serialization format.
            #            WebDataset sees "png" and automatically calls
            #            img.save(buffer, format="PNG") on the PIL Image.
            #            Other supported formats: "jpg", "ppm", "npy", "json"
            #
            # "cls":     The class label (integer 0-9). WebDataset serializes
            #            this as a text file containing the string representation.
            #            When reading back, .decode() will parse it back to int.
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

# boto3 is the official AWS SDK for Python.
# endpoint_url overrides the default AWS endpoint to point at our local MinIO.
# This is the ONLY change needed to switch between local dev and production S3.
s3 = boto3.client(
    's3', 
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

# Create the bucket and set a public-read policy so that WebDataset can
# stream data over plain HTTP (without AWS Signature V4 authentication).
# In production, you would use IAM roles and signed URLs instead.
import json
try:
    s3.create_bucket(Bucket="cifar-streaming")
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "PublicRead",
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::cifar-streaming/*"]
        }]
    }
    s3.put_bucket_policy(Bucket="cifar-streaming", Policy=json.dumps(policy))
except:
    pass

# glob.glob returns all files matching the pattern in the current directory.
# Each .tar shard is uploaded as a separate S3 object.
for file in glob.glob("cifar-train-*.tar"):
    print(f"Uploading {file}...")
    s3.upload_file(file, "cifar-streaming", file)
print("Uploads complete!")
''')


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    prepare_data()
