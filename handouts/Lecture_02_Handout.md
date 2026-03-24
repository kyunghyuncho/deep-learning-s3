# Lecture 2 Handout: The Ingestion Bottleneck and Streaming

## Core Concepts

### 1. The Shuffle Buffer

**Problem:** Global shuffling requires $\mathcal{O}(D)$ memory where $D$ is the full dataset size. For a 10 TB dataset, this is infeasible.

**Solution:** A sliding-window buffer of size $K$ provides pseudo-random sampling with only $\mathcal{O}(K)$ memory.

**Algorithm:**
1. Fill a buffer with $K$ items from the sequential stream
2. Randomly pop one item from the buffer → yield it
3. Fill the vacated slot with the next item from the stream
4. Repeat

**Key property for SGD:** Stochastic Gradient Descent requires that consecutive minibatches be sufficiently **decorrelated**, not perfectly uniformly random. A buffer of $K \geq 1000$ is typically sufficient.

**WebDataset provides two levels of shuffling:**

```python
dataset = (
    wds.WebDataset(url, shardshuffle=True)  # Shuffle shard order (inter-shard)
    .shuffle(1000)                           # Buffer shuffle within shards (intra-shard)
)
```

---

### 2. PyTorch DataLoader Optimization

The GPU is the **consumer**; the DataLoader is the **producer**. If the producer is slower, the GPU idles.

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `batch_size` | Samples per gradient update | 64–1024 |
| `num_workers` | Background processes fetching data in parallel | 2–8 |
| `pin_memory` | Page-locked CPU memory for fast GPU transfer | True (CUDA only) |
| `persistent_workers` | Keep worker processes alive between epochs | True |

**Why `num_workers > 0`?** Python's GIL (Global Interpreter Lock) prevents true multi-threading. `num_workers` spawns separate **processes** (via `multiprocessing`), each independently streaming, decoding, and queuing data. The main process just dequeues batches.

**Trade-off:** Too few workers → GPU starves. Too many → CPU contention and excessive memory usage.

```python
loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,
    pin_memory=True
)
```

---

### 3. ResNet Architecture

#### The Vanishing Gradient Problem

In deep networks, gradients are multiplied through $L$ layers during backpropagation:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \prod_{i=1}^{L} \frac{\partial x_i}{\partial x_{i-1}} \cdot \frac{\partial \mathcal{L}}{\partial x_L}$$

If each factor has magnitude $< 1$, gradients shrink exponentially:

| Depth | Gradient magnitude ($0.9^L$) |
|-------|------------------------------|
| 10 | 0.35 |
| 50 | 0.005 |
| 100 | ≈ 0 (vanished) |

#### The Residual Solution (He et al., 2015)

Instead of learning $H(x)$ directly, learn a perturbation $\mathcal{F}(x)$ added to identity:

$$y = \mathcal{F}(x, \{W_i\}) + x$$

Derivative via chain rule:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + \mathbf{1} \right)$$

The **+1 term** guarantees that gradients always have a direct path through the network, even if $\frac{\partial \mathcal{F}}{\partial x} \approx 0$.

#### Our ResNet9 Architecture

| Layer | Operation | Output Shape | Parameters |
|-------|-----------|-------------|------------|
| Input | RGB image | (N, 3, 32, 32) | — |
| Conv + BN + ReLU | Conv2d(3→64, 3×3, pad=1) | (N, 64, 32, 32) | 1,792 |
| Residual Conv 1 | Conv2d(64→64, 3×3, pad=1) + ReLU | (N, 64, 32, 32) | 36,928 |
| Residual Conv 2 | Conv2d(64→64, 3×3, pad=1) | (N, 64, 32, 32) | 36,928 |
| Skip Connection | **F(x) + x**, then ReLU | (N, 64, 32, 32) | 0 |
| MaxPool | MaxPool2d(2) | (N, 64, 16, 16) | 0 |
| Flatten | reshape | (N, 16384) | 0 |
| Classifier | Linear(16384 → 10) | (N, 10) | 163,850 |
| **Total** | | | **~239K** |

---

### 4. Training Fundamentals

#### Categorical Cross-Entropy Loss

The model outputs logits $z \in \mathbb{R}^C$. Softmax converts to probabilities:

$$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

Cross-entropy loss for the true class $t$:

$$\mathcal{L} = -\log(\hat{y}_t) = -z_t + \log\left(\sum_{j=1}^{C} e^{z_j}\right)$$

The second form (log-sum-exp) is used in PyTorch's `F.cross_entropy` for numerical stability — it avoids computing $e^{z_j}$ which can overflow.

#### AdamW Optimizer

Adam with decoupled weight decay. Maintains per-parameter adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment / momentum)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment / variance)}$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update rule with decoupled weight decay:

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

| Hyperparameter | Default | Role |
|----------------|---------|------|
| $\eta$ (lr) | 1e-3 | Step size |
| $\beta_1$ | 0.9 | Momentum decay |
| $\beta_2$ | 0.999 | Variance decay |
| $\epsilon$ | 1e-8 | Numerical stability |
| $\lambda$ | 0.01 | Weight decay (L2 regularization) |

#### Data Layout: NHWC vs NCHW

| Format | Meaning | Used By |
|--------|---------|---------|
| NHWC | Batch × Height × Width × Channels | Image files, TensorFlow default |
| NCHW | Batch × Channels × Height × Width | PyTorch, cuDNN (optimized for GPU) |

WebDataset outputs NHWC. PyTorch Conv2d expects NCHW. We convert with:

```python
images = images.permute(0, 3, 1, 2).contiguous()
```

- `.permute()` rearranges dimensions without copying (changes memory strides)
- `.contiguous()` copies to sequential memory layout (required for backpropagation)

---

### 5. PyTorch Lightning

Lightning separates **research code** from **engineering code**:

| You Write | Lightning Handles |
|-----------|-------------------|
| `training_step()` | `loss.backward()` |
| `test_step()` | `optimizer.step()` and `zero_grad()` |
| `configure_optimizers()` | Device placement (CPU/GPU/TPU) |
| | Gradient clipping |
| | Logging and checkpointing |
| | Multi-GPU / distributed training |

---

## Exercises

1. Run the shuffle buffer simulator with $K = 1$, $K = 100$, $K = 1000$, $K = 5000$. At what value of $K$ does the distribution become approximately uniform? What does this suggest about the memory-accuracy trade-off?

2. Train the ResNet9 for 1 epoch. Record the final `train_acc` and `test_acc`. Is the model overfitting or underfitting? How can you tell?

3. **Challenge:** Modify the `configure_optimizers` method to use plain SGD with momentum instead of AdamW. Does convergence speed change?
