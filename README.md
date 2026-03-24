# Bridging Big Data and Distributed AI Ingestion

This repository serves as the unified lecture material and interactive environment for distributed data ingestion and streaming architectures.

## Lecture Notes & Theoretical Foundations

Modern Deep Learning systems ingest terabytes of data across clustered GPUs. A naive translation of MapReduce paradigms (like Spark writing millions of small parquet files) to deep learning trainers causes critical hardware starvation. Below are the foundational limits and solutions to these bottlenecks.

### Lecture 1: The Storage and Serialization Bridge

**The Small Files Problem**
Cloud object stores (like S3) and standard hard drives incur severe seek-time penalties when resolving files. When a machine learning algorithm demands millions of individual images independently over a network, the overhead of establishing connections dominates throughput.

Let $N$ be the total dataset size, $S_{file}$ be the average file size, $B_{network}$ the global network bandwidth, and $T_{seek}$ the Time-To-First-Byte latency of the storage system.

The baseline equation for reading independent files is bounded by:
$$T_{total} = N \times \left(T_{seek} + \frac{S_{file}}{B_{network}}\right)$$

If $T_{seek}$ is massive (e.g., standard HTTP roundtrips), GPUs stay idle at 0% utilization. To counteract this, datasets are pre-processed and serialized into contiguous shards (e.g., $1000$ images per `.tar` file) using architectures like `WebDataset`. The GPUs sequentially read bytes out of the stream rather than opening files:

$$T_{total} \approx T_{seek\_archive} + \frac{N \times S_{file}}{B_{network}}$$

By sharding data, we amortize $T_{seek}$ across thousands of items, effectively dissolving the seek penalty.

### Lecture 2: The Ingestion Bottleneck and Streaming

**The Shuffle Buffer Mechanism**
A globally uniform dataset shuffle across a remote 10TB S3 bucket is computationally unfeasible. True uniform random sampling across the data size $D$ requires $\mathcal{O}(D)$ memory.
To solve this, deep learning stream consumers employ a sliding-window buffer. If the buffer size is $K$, we fill $K$ items into RAM, and randomly select one to yield, immediately fetching the next stream object to replace it. This uses $\mathcal{O}(K)$ memory, generating pseudo-randomness sufficient for Stochastic Gradient Descent (SGD) while preventing memory overflow.

**Deep Learning and Residual Layers**
In this coursework, we simulate ingestion flows directly into a `ResNet9` neural network topology. Adding depth to networks can cause vanishing gradients. The mathematical innovation of ResNet is the Residual Block:
$$y = \mathcal{F}(x, \{W_i\}) + x$$

When running back-propagation, the derivative guarantees a healthy gradient signal is sent back to earlier layers linearly:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)$$

This architecture is optimized against Categorical Cross-Entropy loss mapping the true probability $y_c$ and predicted probability $$\hat{y}_c$$:
$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

### Lecture 3: Distributed Paradigms

**Parameter Server vs Ring-AllReduce**
When parallelizing training, neural network parameter gradients must be aggregated across node boundaries. Legacy Big Data clusters (Hadoop/Spark) use TCP/IP. Modern nodes utilize NVLink or InfiniBand over toruses to accomplish remote direct memory access (RDMA).

*Parameter Server (Spark-Style Hub-and-Spoke):*
Given $M$ model parameters, $N$ nodes, and bandwidth $B$, if a centralized server must ingest from every worker simultaneously, a network bottleneck occurs:
$$T_{PS} = \frac{M \cdot N}{B}$$
As $N$ scales, $T_{PS}$ scales linearly toward infinity; the system grinds to a halt.

*Ring-AllReduce (Collective Communication):*
Nodes are structured logically in a ring pipeline. The algorithm operates in two phases (Scatter-Reduce and All-Gather). Each executes $N-1$ steps, processing $\frac{M}{N}$ bytes locally on the GPU bus.
$$T_{Ring} = 2 \times \frac{M(N-1)}{N \cdot B}$$
Taking the horizontal limit as node counts diverge ($N \to \infty$):
$$T_{Ring} \approx \frac{2M}{B}$$
The time strictly plateaus, entirely circumventing the centralized bandwidth limit.

---

## Setup Instructions

### Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **uv** (for fast Python environment management)

### 1. Initialize Virtual Environment
Initialize your local isolated Python environment and inject all underlying tensor operation libraries.
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Start the Local S3 (MinIO) Datacenter
Run the MinIO server script to dynamically fetch your OS's standalone storage binary and synthesize an Amazon S3 endpoint over port 9000.
```bash
python scripts/00_start_minio.py
```
*Note: Ensure this process remains running in a background terminal. You can observe the console at `http://127.0.0.1:9000` via `minioadmin` parameters. A local `cifar-streaming` bucket must exist.*

### 3. Serialize Data to Shards
Execute the preprocessing stream to compress CIFAR-10 images into raw byte representations via `.tar` archives in batches of 1000 items.
```bash
python scripts/01_prepare_shards.py
```
*Follow the printed setup instructions to deploy the generated shards straight to the localhost MinIO bucket via boto3.*

### 4. Application Consumer
Boot the Jupyter framework to sequentially work through the interactive `ipywidgets` simulations and training loops mapping our underlying theory to practice:
```bash
jupyter notebook notebooks/
```
