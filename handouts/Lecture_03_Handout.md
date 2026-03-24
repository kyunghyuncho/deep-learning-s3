# Lecture 3 Handout: Distributed Paradigms

## Core Concepts

### 1. The Gradient Synchronization Problem

In data-parallel distributed training, each GPU:
1. Holds a **full copy** of the model
2. Processes a **different minibatch**
3. Computes **local gradients** via backpropagation
4. **Synchronizes** gradients with all other GPUs (the bottleneck)
5. Updates its local model with the averaged gradient

The synchronization step must exchange the full model's gradients every iteration (~0.5–2 seconds):

| Model | Parameters | Gradient Size (FP32) |
|-------|-----------|---------------------|
| ResNet-50 | 25.6M | 97.7 MB |
| BERT-Large | 340M | 1.3 GB |
| GPT-2 | 1.5B | 5.7 GB |
| GPT-3 | 175B | 700 GB |
| Llama 3 70B | 70B | 280 GB |

*FP32 = 4 bytes per parameter. Mixed precision (FP16/BF16) halves the gradient size.*

---

### 2. Hardware Primitives

#### Standard Ethernet (Spark/Hadoop Clusters)

**Data path:** GPU → PCIe Bus → CPU → NIC → Ethernet Switch → NIC → CPU → PCIe Bus → GPU

- CPU manages every packet (interrupt-driven or polling)
- Full TCP/IP stack: ~10–100 μs latency per packet
- Maximum practical throughput: ~12.5 GB/s (100 Gbps Ethernet)
- The CPU becomes the bottleneck, not the network

#### InfiniBand / NVLink (AI Clusters)

**Data path:** GPU → NVLink → GPU (direct, no CPU)

The key technology is **RDMA** — Remote Direct Memory Access:

| Feature | TCP/IP | RDMA |
|---------|--------|------|
| CPU involvement | Every packet | **Zero** (kernel bypass) |
| Memory copies | User → kernel → NIC | **Zero-copy** (DMA engine) |
| Latency | ~10–100 μs | **~1 μs** |
| CPU utilization | High | **Near zero** |

**Bandwidth comparison:**

| Interconnect | Bandwidth | Typical Use |
|-------------|-----------|-------------|
| 1 Gbps Ethernet | 0.125 GB/s | Office networks |
| 100 Gbps Ethernet | 12.5 GB/s | Spark clusters |
| InfiniBand NDR | 50 GB/s | HPC clusters |
| NVLink 4.0 | 900 GB/s | Intra-node GPU mesh |

---

### 3. Parameter Server (Hub-and-Spoke)

The parameter server pattern mirrors Spark's driver-worker architecture:

1. Each worker sends its gradient ($M$ bytes) to one central server
2. The server averages all gradients
3. The server broadcasts the result back

**Communication time:**

$$T_{PS} = \frac{M \cdot N}{B}$$

where $M$ = model size, $N$ = number of nodes, $B$ = server NIC bandwidth.

**Why it fails:** The server's NIC must handle $M \times N$ total bytes. As $N$ grows, time grows linearly — a hard physical limit.

**Numerical examples** (model: 5.7 GB / GPT-2, bandwidth: 50 GB/s):

| Nodes (N) | T_PS | Practical? |
|-----------|------|------------|
| 8 | 0.91s | ✅ Acceptable |
| 64 | 7.3s | ❌ Too slow |
| 1024 | 117s | ❌ Completely unusable |

---

### 4. Ring-AllReduce (Collective Communication)

#### Algorithm

Arrange $N$ nodes in a logical ring. Divide the gradient into $N$ chunks.

**Phase 1 — Scatter-Reduce** ($N-1$ steps):
- Each node sends one $\frac{M}{N}$ chunk to its right neighbor
- Upon receiving, **sum** the chunk with the local copy
- After $N-1$ steps, each node holds the **complete sum** of exactly one chunk

**Phase 2 — All-Gather** ($N-1$ steps):
- Each node forwards its completed chunk to the right
- After $N-1$ steps, every node has the full, fully-reduced gradient

#### Communication Time

Total data sent per node: $2 \times (N-1) \times \frac{M}{N}$ bytes

$$T_{Ring} = \frac{2 \cdot M \cdot (N-1)}{N \cdot B}$$

**Asymptotic behavior:**

$$\lim_{N \to \infty} T_{Ring} = \frac{2M}{B} \quad \text{(constant — independent of N!)}$$

#### Bandwidth Optimality

Ring-AllReduce is **bandwidth-optimal**: no collective algorithm can achieve lower total communication.

**Proof sketch:** Every node must contribute its $M$ bytes (scatter) and receive the final $M$ bytes (gather). Therefore, each node must transfer at least $2M$ bytes total. Ring-AllReduce achieves exactly $\frac{2M(N-1)}{N} \to 2M$ as $N \to \infty$.

**Bandwidth utilization:**

$$\text{Utilization}_{PS} = \frac{1}{N} \xrightarrow{N \to \infty} 0\%$$

$$\text{Utilization}_{Ring} = \frac{N-1}{N} \xrightarrow{N \to \infty} 100\%$$

---

### 5. Head-to-Head Comparison

| Metric | Parameter Server | Ring-AllReduce |
|--------|-----------------|----------------|
| Time complexity | $\mathcal{O}\!\left(\frac{MN}{B}\right)$ | $\mathcal{O}\!\left(\frac{M}{B}\right)$ |
| Scales with N? | ❌ Linear slowdown | ✅ Constant time |
| Server bottleneck | Yes (single NIC) | None (symmetric) |
| BW utilization | $1/N → 0\%$ | $(N{-}1)/N → 100\%$ |
| Implementation | Simple | Requires ring topology |
| Fault tolerance | Server is SPOF | Any node failure breaks ring |

**GPT-2 (5.7 GB) at scale, B = 50 GB/s:**

| Nodes | T_PS | T_Ring | Speedup |
|-------|------|--------|---------|
| 8 | 0.91s | 0.20s | 4.6× |
| 64 | 7.3s | 0.22s | 33× |
| 256 | 29.2s | 0.23s | 127× |
| 1024 | 117s | 0.23s | 509× |
| 8192 | 934s | 0.23s | 4,061× |

---

### 6. Modern Topologies

Real GPU clusters use generalizations of Ring-AllReduce:

| Topology | Used In | Key Idea |
|----------|---------|----------|
| **2D/3D Torus** | Google TPU Pods | Multiple simultaneous rings across dimensions |
| **Fat Tree** | InfiniBand clusters | Full bisection bandwidth via hierarchical switches |
| **NVSwitch Mesh** | NVIDIA DGX (intra-node) | All-to-all at full NVLink speed (8 GPUs) |
| **Rail-Optimized** | NVIDIA SuperPODs | Minimizes inter-rail (inter-switch) hops |

**Common principle:** All advanced topologies ensure that every node can send and receive at full bandwidth simultaneously — the same property that makes Ring-AllReduce bandwidth-optimal.

#### NCCL (NVIDIA Collective Communications Library)

In practice, PyTorch uses NCCL to automatically:
1. Detect the hardware topology (NVLink, InfiniBand, Ethernet)
2. Select the optimal algorithm (ring, tree, or hybrid)
3. Overlap communication with computation (pipeline parallelism)

```python
# PyTorch DDP automatically uses NCCL Ring-AllReduce
model = DistributedDataParallel(model)
```

---

## Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $T_{PS} = \frac{M \cdot N}{B}$ | Parameter server time (linear in N) |
| $T_{Ring} = \frac{2M(N{-}1)}{NB}$ | Ring-AllReduce time |
| $\lim_{N \to \infty} T_{Ring} = \frac{2M}{B}$ | Ring asymptote (constant) |
| $\text{Util}_{PS} = \frac{1}{N}$ | PS bandwidth utilization |
| $\text{Util}_{Ring} = \frac{N{-}1}{N}$ | Ring bandwidth utilization |

---

## Exercises

1. Using the interactive simulator, configure a 30,000-node cluster with a 10 GB model. What is the Ring-AllReduce communication time? What would the Parameter Server time be?

2. A company wants to train a 70B-parameter model (280 GB gradients) on 1024 GPUs connected by InfiniBand NDR (50 GB/s). Calculate $T_{Ring}$ and determine if this is feasible within a 2-second iteration budget.

3. **Discussion:** Ring-AllReduce breaks if any single node fails. How would you design a fault-tolerant collective communication algorithm? What trade-offs would you make?

4. **Challenge:** Derive why a 2D torus with $\sqrt{N} \times \sqrt{N}$ nodes can run two simultaneous rings, achieving $T_{2D} \approx \frac{2M}{\sqrt{N} \cdot B}$. Under what conditions does this beat a 1D ring?
