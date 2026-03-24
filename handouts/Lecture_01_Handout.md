# Lecture 1 Handout: The Storage and Serialization Bridge

## Core Concepts

### 1. The Small Files Problem

When training data is stored as millions of individual files, each file access incurs a **seek penalty** — the time to locate, authenticate, and open the file before any data bytes are transferred.

| Component | Typical Latency |
|-----------|----------------|
| DNS Resolution | 1–50 ms |
| TCP Handshake (SYN/SYN-ACK/ACK) | 1–10 ms |
| TLS Negotiation (HTTPS) | 5–30 ms |
| S3 API Authentication (AWS Sig V4) | 5–20 ms |
| Disk/SSD Seek | 0.1–10 ms |
| **Total T_seek** | **~12–120 ms** |

**Independent files latency:**

$$T_{total} = N \times \left(T_{seek} + \frac{S_{file}}{B_{network}}\right)$$

**Example (CIFAR-10):** $N = 60{,}000$, $T_{seek} = 50\text{ms}$, $S_{file} = 3\text{KB}$, $B = 1\text{Gbps}$

- Seek overhead alone: $60{,}000 \times 0.05\text{s} = 3{,}000\text{s}$ (50 minutes of GPU idle time)
- Transfer time: $60{,}000 \times \frac{3 \times 10^{-3}\text{MB}}{125\text{MB/s}} = 1.44\text{s}$
- **Seek dominates by 2000×!**

### 2. Contiguous Sharding

By packing thousands of samples into sequential `.tar` archives, we pay the seek cost once per shard:

$$T_{total} \approx T_{seek\_archive} + \frac{N \times S_{file}}{B_{network}}$$

With 1,000 items per shard: 50 shards × $T_{seek}$ = 2.5s (vs. 3,000s)

---

## Object Storage vs Filesystems

| Feature | Filesystem (HDFS, ext4) | Object Storage (S3) |
|---------|------------------------|---------------------|
| Namespace | Hierarchical directories | Flat (buckets + keys) |
| Mutability | Read/write/append | Immutable (replace only) |
| API | POSIX (open/read/seek/close) | HTTP (GET/PUT/DELETE) |
| Scale | ~millions of files | **Trillions** of objects |
| Access | Local/NFS mount | Network-native (HTTP) |

### Key S3 Terminology

- **Bucket**: Top-level container (like a database schema)
- **Object**: A file + metadata, identified by a string key
- **Pagination**: List responses limited to 1,000 objects per page
- **IAM Credentials**: Access key ID + secret access key (like username/password)
- **Bucket Policy**: JSON document controlling access permissions

---

## WebDataset Format

### Why `.tar` and Not `.zip`?

| Property | ZIP | TAR |
|----------|-----|-----|
| Index location | End of file (central directory) | No index needed |
| Reading strategy | Must seek to EOF first | Stream from byte 0 |
| HTTP streaming | ❌ Not possible | ✅ Perfectly streamable |
| Compression | Per-file | Not built-in (use `.tar.gz`) |

### Sample Structure Inside a Shard

```
cifar-train-000000.tar
├── 000000.png    (image bytes)
├── 000000.cls    (class label as text)
├── 000001.png
├── 000001.cls
├── ...
└── 000999.cls
```

Files sharing the same key prefix (e.g., `000042`) are grouped as one training sample.

### Pipeline Stages

```python
dataset = (
    wds.WebDataset(url, shardshuffle=False)   # Stream tar bytes over HTTP
    .decode("rgb")                             # PNG → float32 (H,W,3) [0,1]
    .to_tuple("png", "cls")                    # → (image, label) tuples
)
```

| Stage | Input | Output |
|-------|-------|--------|
| `WebDataset(url)` | URL pattern with brace expansion | Raw tar byte streams |
| `.decode("rgb")` | PNG byte buffers | numpy float32 arrays (H,W,3) |
| `.to_tuple("png", "cls")` | Dict of decoded fields | (image, label) tuples |
| `.shuffle(K)` | Ordered stream | Pseudo-random stream |

---

## Lab Setup Reference

```bash
# 1. Start MinIO S3 server
python scripts/00_start_minio.py

# 2. Generate and upload shards (in another terminal)
python scripts/01_prepare_shards.py

# 3. Launch Jupyter
jupyter lab
```

**MinIO Console:** http://127.0.0.1:9000 — Credentials: `minioadmin` / `minioadmin`

---

## Exercises

1. In the I/O Latency Simulator, find the file size at which transfer time equals seek time for standard HTTP access (~50ms seek). What does this tell you about when sharding matters most?

2. Using `boto3`, list the objects in your bucket. Verify each shard is ~3 MB. Why is this a good shard size?

3. Stream 100 images from MinIO using WebDataset. Do any fail to decode? What does this tell you about the tar format's resilience?
