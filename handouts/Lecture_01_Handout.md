# Lecture 1 Handout: The Storage and Serialization Bridge

## Introduction: The Gap Between Big Data and AI

In modern machine learning pipelines, a fundamental tension exists between the systems that prepare data and the systems that consume it. Data engineering frameworks like Apache Spark or Hadoop are designed to process massive, distributed datasets. They optimize for parallel throughput by splitting outputs into hundreds of thousands of independent files across a cluster. Deep learning frameworks like PyTorch, however, run on Graphics Processing Units (GPUs) that consume training data at blistering speeds. When an AI pipeline attempts to read these millions of small, distributed files directly, a severe bottleneck emerges. This lecture explores the root of this "Small Files Problem," introduces Object Storage as a scalable alternative to traditional filesystems, and demonstrates how to build high-throughput serialization and streaming pipelines to keep GPUs fully utilized.

---

## 1. The Small Files Problem

When training data is stored as millions of individual files (e.g., one JPEG file per image in a dataset), a severe performance penalty occurs. Every single file access incurs a combination of network, authentication, and hardware latencies before a single byte of actual data is transferred. This absolute time cost is known as the **seek penalty**. 

Because Spark outputs partitions as files, an AI training job reading these partitions over a network must pay this seek penalty for every single sample.

| Component | Typical Latency | Description |
|-----------|----------------|-------------|
| DNS Resolution | 1–50 ms | Locating the server IP address |
| TCP Handshake | 1–10 ms | Establishing the network connection (SYN/ACK) |
| TLS Negotiation | 5–30 ms | Setting up encryption for an HTTPS connection |
| S3 API Auth | 5–20 ms | Verifying IAM credentials (AWS Sig V4) |
| Disk/SSD Seek | 0.1–10 ms | Finding the physical block on the storage medium |
| **Total $T_{seek}$** | **~12–120 ms** | **Paid per file, regardless of file size** |

### The I/O Latency Equation

We can mathematically formalize the time it takes to read a dataset of $N$ independent files. The total time ($T_{total}$) is the sum of the seek penalties plus the time to actually transfer the bytes over the network bandwidth ($B_{network}$):

$$T_{total} = N \times \left(T_{seek} + \frac{S_{file}}{B_{network}}\right)$$

**A Concrete Example: CIFAR-10**
Let's apply this to the CIFAR-10 dataset. Suppose we have $N = 60{,}000$ images, where each image is a tiny $S_{file} = 3\text{ KB}$. Assume a very fast local network of $B = 1\text{ Gbps}$ (125 MB/s), and an optimistic seek latency of $T_{seek} = 50\text{ ms}$ per file.

- Transfer time: $60{,}000 \times \frac{3 \times 10^{-3}\text{MB}}{125\text{MB/s}} = 1.44\text{ seconds}$
- Seek overhead: $60{,}000 \times 0.05\text{s} = 3{,}000\text{ seconds}$

Even on a fast network, transferring the actual image data takes less than 2 seconds, but establishing the connections to read those 60,000 files takes **50 minutes**. During this time, your expensive GPU is doing nothing but waiting for data. The seek penalty dominates the actual data transfer by a factor of 2,000!

---

## 2. The Solution: Contiguous Sharding

To solve this, we must amortize (spread out) the cost of the seek penalty. We achieve this through **contiguous sharding**. By packing thousands of samples sequentially into a single large archive file (a "shard"), we only pay the network seek penalty once per shard rather than once per sample. The equation transforms:

$$T_{total} \approx T_{seek\_archive} + \frac{N \times S_{file}}{B_{network}}$$

If we pack 1,000 images into a single `.tar` archive, we reduce 60,000 individual files to just 50 shards. We now only pay the seek penalty 50 times. 

- $50 \times 0.05\text{s} = \mathbf{2.5\text{ seconds}}$ total seek latency (down from 3,000 seconds).

The GPU now spends almost 100% of its time training on data, rather than idling.

---

## 3. Object Storage vs Traditional Filesystems

Where do we store these shards? While HDFS or local network drives are common in traditional enterprise IT, the AI industry has overwhelmingly standardized on **Object Storage** (specifically the Amazon S3 protocol) for managing massive datasets.

Object storage Abandons the concept of hierarchical directories and file system tree traversal. Instead, it places data into a flat, infinitely scalable namespace.

| Feature | Filesystem (HDFS, ext4) | Object Storage (S3) |
|---------|------------------------|---------------------|
| Namespace | Hierarchical folders/directories | Flat namespace (buckets + string keys) |
| Mutability | Read/write/append | Immutable (rewrite entire object to modify) |
| API | POSIX strict semantics (open/seek) | Stateless HTTP (GET/PUT/DELETE) |
| Scale limits | inode exhaustion (~millions of files) | Effectively limitless (**Trillions** of objects) |
| Access | Typically requires local OS mounting | Network-native, reachable from anywhere |

### Core S3 Terminology
When interacting with an Object Store like AWS S3 or MinIO, you must understand a specific vocabulary:
- **Bucket**: The top-level namespace container. Think of it like a database schema or a root drive.
- **Object**: The fundamental unit of storage. An object consists of the file data (the bytes) and associated metadata.
- **Key**: The unique string identifier for an object in a bucket. While keys often look like file paths (e.g., `data/train/001.tar`), there are no actual folders; the slashes are just characters in the string.
- **Pagination**: Because buckets can contain billions of objects, querying a bucket's contents never returns the full list. Instead, the API typically returns a maximum of 1,000 objects per page, requiring you to ask for a "continuation token" to get the next batch.
- **IAM Credentials**: Access requires a pair of keys: an `Access Key ID` (which acts as a username) and a `Secret Access Key` (which acts as an API password).

---

## 4. WebDataset Serialization Architecture

To implement our contiguous sharding strategy, we need a specific file format. The `WebDataset` library provides a standardized way to package and stream PyTorch datasets using POSIX tar archives. 

### Why Use `.tar` Instead of `.zip`?

It might seem intuitive to zip the data, but the ZIP format is fundamentally incompatible with pure network streaming. 

A ZIP file writes its "Central Directory" (the index of where files are located) at the very end of the file. To read any file in a ZIP, a program must first seek to the end, read the index, and then jump back to the specific byte offset. Over HTTP, this requires multiple range requests and random access jumps.

A TAR file, conversely, is simply concatenated data. It writes a small header, followed by the file bytes, followed by the next header. It contains no central index. This means a program can open an HTTP stream and immediately begin reading and decoding files from byte zero without knowing how large the archive is or seeking forward. This makes TAR files perfectly streamable over stateless HTTP connections.

### Organizing Samples Inside a Shard

In deep learning, a single "sample" often consists of multiple related files (e.g., an image and its corresponding label or mask). WebDataset organizes these by grouping files that share a common prefix (the "key"). 

```text
cifar-train-000000.tar
├── 000000.png    (The image input)
├── 000000.cls    (The label, stored as text)
├── 000001.png
├── 000001.cls
└── ...
```
When iterating over this tar stream, WebDataset collects all files belonging to `000000` into a single Python dictionary before yielding the sample to the DataLoader.

### The Pipeline Architecture

A WebDataset pipeline is built using a functional, chained API. Importantly, because it reads from a continuous stream, it is **lazy**. It never materializes the full dataset into RAM or local disk.

```python
dataset = (
    wds.WebDataset(url, shardshuffle=False)   # 1. Open HTTP byte stream
    .decode("rgb")                             # 2. Decompress bytes to numpy arrays
    .to_tuple("png", "cls")                    # 3. Restructure dicts into PyTorch tuples
)
```

1. **`WebDataset(url)`**: Takes a URL (often with brace expansion to specify multiple shards) and yields raw byte streams.
2. **`.decode("rgb")`**: Automatically detects image extensions (like `.png`) and decodes the byte buffers into standard float32 numpy arrays scaled to `[0, 1]`.
3. **`.to_tuple("png", "cls")`**: Extracts the specific fields requested from the dictionary, yielding the standard `(x, y)` tuple that PyTorch training loops expect.

---

## 5. Lab Environment and Execution

In the companion Jupyter notebook, you will use MinIO, an open-source object storage server that implements the exact same API as AWS S3. This allows us to write enterprise-grade cloud code and test it entirely on a local laptop without requiring cloud credentials or incurring egress charges.

To prepare your environment:

```bash
# Terminal 1: Start the local MinIO server
# This script automatically downloads the binary for your OS and launches it
python scripts/00_start_minio.py

# Terminal 2: Connect to MinIO, download CIFAR-10, and serialize it to shards
python scripts/01_prepare_shards.py

# Terminal 3: Launch your notebook environment
jupyter lab
```

You can view the objects you created by opening the MinIO Web Console at `http://127.0.0.1:9000` using the username `minioadmin` and password `minioadmin`.

---

## 6. Review Exercises

1. **Analytical:** In the notebook's I/O Latency Simulator, increase the network bandwidth to 100 Gbps. Then, find the "crossover point" regarding file size where the transfer time begins to equal the seek time for standard HTTP access (~50ms seek). What does this tell you about when sharding matters most (e.g., for NLP text vs High-Res Video)?

2. **Practical:** In a Python script, use the `boto3` library to programmatically connect to the MinIO server and list the objects in the bucket you created. Verify that each shard is approximately 3 MB in size. Why might 3 MB be a good size for testing, and what size might you choose for production?

3. **Resilience:** If a `.tar` shard gets corrupted midway through network transmission, a `.zip` file would fail completely. Write a loop to stream 100 images from MinIO using WebDataset. If you were to intentionally corrupt byte 50,000 of the tar file, how does the stream behave? What does this tell you about the tar format's robustness in distributed systems?
