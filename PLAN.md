# PLAN: Bridging Big Data and Distributed AI Ingestion
**Target Course:** CSCI-GA.2437 (Big Data)
**Focus:** Data formatting, S3-compatible streaming, shuffle buffers, and distributed I/O.
**Lecture Duration:** 3 x 100-minute sessions.

## 0. Repository Architecture
The Antigravity agent must generate the following structure with fully functional, documented code:

```text
├── README.md                           # Course instructions, Docker/Python setup, and prerequisites
├── requirements.txt                    # Exact versions: boto3, webdataset, torch, torchvision, pytorch-lightning, ipywidgets, matplotlib, requests
├── scripts/
│   ├── 00_start_minio.py               # Auto-downloads, configures, and runs MinIO locally via subprocess
│   └── 01_prepare_shards.py            # Downloads CIFAR-10, applies transforms, and writes WebDataset .tar shards
└── notebooks/
    ├── Lecture_01_Storage_and_Serialization.ipynb
    ├── Lecture_02_Streaming_and_Ingestion.ipynb
    └── Lecture_03_Distributed_Paradigms.ipynb
```

---

## Lecture 1: The Storage and Serialization Bridge (100 Minutes)
**Core Concept:** Transitioning from Spark's HDFS/Parquet output generation to AI's S3/Sequential-read ingestion.

### 1.1 Infrastructure: The Local S3 Environment
* **LLM Implementation Directive (`00_start_minio.py`):** * Import `platform`, `urllib.request`, `os`, and `subprocess`.
  * Write branching logic: If `system == 'Darwin'` and `machine == 'arm64'`, fetch the Apple Silicon MinIO binary. Map x86/Windows equivalents.
  * Execute `chmod +x` equivalent, then launch via `subprocess.Popen` binding to `127.0.0.1:9000` with `MINIO_ROOT_USER=minioadmin` and `MINIO_ROOT_PASSWORD=minioadmin`.
* **Student Expectation:** Execute the script. Open Microsoft Edge or their preferred browser to `localhost:9000`. Authenticate, navigate the UI, and manually construct a bucket named `cifar-streaming` to internalize the concept of object storage buckets versus standard hierarchical folders.

### 1.2 Data Serialization: Escaping the Small Files Problem
* **Concept:** Spark computes across distributed partitions; AI ingestion starves if fed millions of small, fragmented files.
* **LLM Implementation Directive (`01_prepare_shards.py`):** * Instantiate `torchvision.datasets.CIFAR10`.
  * Open a `webdataset.TarWriter` context manager with a pattern like `cifar-train-%06d.tar`.
  * Iterate over the dataset, writing dictionaries containing `__key__`, `png` (byte array), and `cls` (integer) to the tarball. Chunk files at 1000 samples per shard.
* **The Math (Theory):** The notebook must render the I/O latency equation in LaTeX.
  * Baseline equation for independent files: 
    $$T_{total} = N \times \left(T_{seek} + \frac{S_{file}}{B_{network}}\right)$$
  * Explain that with $N=60000$ and $T_{seek}$ dominating network latency, GPUs idle at 0%.
  * Amortized equation for contiguous sharding (Parquet/Tar):
    $$T_{total} \approx T_{seek\_archive} + \frac{N \times S_{file}}{B_{network}}$$
* **Student Expectation:** Run the serialization script. Write a 10-line Python block using `boto3.client('s3', endpoint_url='http://localhost:9000')` to programmatically upload the generated `cifar-train-*.tar` files to their `cifar-streaming` bucket.

### 1.3 Interactive S3 Exploration
* **LLM Implementation Directive (`Lecture_01_Storage_and_Serialization.ipynb`):** * Write a cell demonstrating `boto3` pagination to list bucket objects.
  * Construct a `wds.WebDataset` pipeline reading directly from the `http://localhost:9000/cifar-streaming/...` endpoint. Include `.decode("rgb")` and `.to_tuple("png", "cls")`.
* **Student Expectation:** Students will execute the pipeline and fetch a single minibatch without touching local SSD storage. They must write a `matplotlib.pyplot.subplots` grid (4x4) to visualize 16 streamed tensors, confirming color channels and image integrity.

---

## Lecture 2: The Ingestion Bottleneck and Streaming (100 Minutes)
**Core Concept:** Building a high-throughput, infinitely streaming DataLoader that mimics enterprise-scale AI training.

### 2.1 The Shuffle Buffer Mechanism
* **Concept:** A global Spark `reduceByKey` or `join` across a remote 10TB S3 bucket is computationally unfeasible.
* **LLM Implementation Directive (`Lecture_02_Streaming_and_Ingestion.ipynb`):** * Implement the `wds.shuffle(initial=1000)` node in the WebDataset pipeline.
* **The Math (Theory):** * Let dataset size be $D$ and buffer size be $K$. True uniform random sampling requires $\mathcal{O}(D)$ memory.
  * Sliding-window buffer shuffling reduces this to $\mathcal{O}(K)$ memory, yielding pseudo-randomness sufficient for SGD.
* **Student Expectation:** Students will modify the buffer parameter $K$ from 10 to 5000. They will plot a histogram of class distributions across 5 sequential minibatches to empirically observe how small buffers lead to highly correlated, high-variance class distributions.

### 2.2 Optimizing the PyTorch DataLoader
* **Concept:** The neural network is merely a consumer; the data engineer's job is to guarantee it is fed fast enough.
* **LLM Implementation Directive:** * Wrap the `wds.WebDataset` in a `torch.utils.data.DataLoader`.
  * Explicitly configure and comment on `batch_size=256`, `num_workers=4`, and `pin_memory=True`.
* **Student Expectation:** Students will write a benchmark loop containing `time.sleep(0.01)` to simulate forward/backward passes. They will iterate over the DataLoader, calculate `Samples / Second`, and tune `num_workers` to find the exact saturation point of their local MinIO network loopback.

### 2.3 [OPTIONAL] Model Architecture Deep Dive: Small ResNet
* **Concept:** Transitioning from the theoretical MapReduce architectures to deep learning topologies.
* **LLM Implementation Directive:** * Define `class ResNet9(nn.Module):` using standard `Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d` blocks. Include exactly one residual addition.
* **The Math (Theory):** * Define the residual mapping:
    $$y = \mathcal{F}(x, \{W_i\}) + x$$
  * Show the derivative to explain gradient flow bypassing the non-linear block:
    $$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)$$
* **Student Expectation:** Review the forward pass code to understand how tensors change shape.

### 2.4 The Consumer: PyTorch Lightning Training Loop
* **Concept:** Standardizing the training loop to abstract away GPU casting and optimizer stepping.
* **LLM Implementation Directive:** * Write a `LightningModule` containing `training_step` (computing `F.cross_entropy`) and `configure_optimizers` (returning `torch.optim.AdamW`).
  * Render the categorical cross-entropy formula:
    $$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$
* **Student Expectation:** Instantiate `pl.Trainer(max_epochs=2, ...)` and execute `.fit()`. Students will use the `%tensorboard --logdir lightning_logs` magic command to interactively monitor the loss curve.

---

## Lecture 3: Distributed Paradigms (100 Minutes)
**Core Concept:** Contrasting Big Data MapReduce aggregation networks with AI Collective Communication topologies (NCCL).

### 3.1 Network Topology and Bandwidth
* **Concept:** Why Spark fails at parameter aggregation.
* **LLM Implementation Directive (`Lecture_03_Distributed_Paradigms.ipynb`):** * Provide a textual breakdown of standard Ethernet (TCP/IP overhead, switches) versus InfiniBand/NVLink (direct memory access, synchronized rings).

### 3.2 The Parameter Server (Spark-Style) vs. Ring-AllReduce
* **The Math (Theory):**
  * **Parameter Server:** Let $M$ be model size (bytes), $N$ be nodes, $B$ be bandwidth. 
    * The server must receive $M \times N$ bytes. 
    * Time bottleneck:
      $$T_{PS} = \frac{M \cdot N}{B}$$
  * **Ring-AllReduce:** * Scatter-reduce: $N-1$ steps, each transferring $\frac{M}{N}$ bytes.
    * All-gather: $N-1$ steps, each transferring $\frac{M}{N}$ bytes.
    * Total communication time:
      $$T_{Ring} = 2 \times \frac{M(N-1)}{N \cdot B}$$
    * Show the limit: As $N \to \infty$, $T_{Ring} \approx \frac{2M}{B}$.

### 3.3 Interactive Simulation
* **LLM Implementation Directive:** * Use `ipywidgets.interactive` or `VBox`/`HBox` layouts.
  * Create `IntSlider` for $N \in [2, 30000]$ (log scale preferred), `FloatSlider` for $M \in [0.1, 100]$ GB, and `FloatSlider` for $B \in [1, 400]$ GB/s.
  * Write an update function that calculates $T_{PS}$ and $T_{Ring}$ arrays and updates a `matplotlib.pyplot` line chart dynamically.
* **Student Expectation:** Students will manipulate the sliders to simulate a 30,000 GPU cluster. They will write a markdown cell explaining why the $T_{PS}$ line approaches vertical infinity while $T_{Ring}$ plateaus, solidifying their understanding of why modern GPU clusters are built as toruses rather than hub-and-spoke networks.
