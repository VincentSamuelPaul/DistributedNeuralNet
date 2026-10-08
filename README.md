# Distributed Neural Network

A from-scratch distributed deep learning system that trains a feedforward neural network on MNIST across multiple Docker containers, with a live digit-recognition demo and real-time training metrics.

**Live demo:** [distributedneuralnet.vincents.systems](https://distributedneuralnet.vincents.systems)

---

## What it does

Four worker containers each train on a quarter of the MNIST dataset simultaneously. After every gradient step, they send compressed gradient updates to a central Go parameter server, which averages them and returns the result — the core idea behind distributed synchronous SGD. Workers also send their full weights after each epoch so the server can checkpoint and keep everyone in sync. A Flask inference server watches that checkpoint and hot-reloads the model, feeding a React frontend where you can draw a digit and see the network's prediction in real time.

**Accuracy:** ~96.2% after ~198 distributed epochs (~30s/epoch). The single-node baseline reaches ~97.6% by epoch 40 with the same architecture.

---

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Worker 0   │   │  Worker 1   │   │  Worker 2   │   │  Worker 3   │
│  MNIST[0/4] │   │  MNIST[1/4] │   │  MNIST[2/4] │   │  MNIST[3/4] │
│  port 8000  │   │  port 8001  │   │  port 8002  │   │  port 8003  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │   gRPC gradients (Top-K compressed)│                 │
       └──────────────────┴────────┬────────┴─────────────────┘
                                   │
                          ┌────────▼────────┐
                          │ Parameter Server │  Go + gRPC
                          │   port 50051    │  averages gradients
                          │                 │  saves checkpoint
                          └────────┬────────┘
                                   │ checkpoint.json
                          ┌────────▼────────┐
                          │ Inference Server │  Flask (Python)
                          │   port 5000     │  hot-reloads model
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │    Frontend     │  React (no build step)
                          │  28×28 canvas   │  confidence bar chart
                          └─────────────────┘

        Prometheus (9090) + Grafana (3000) scrape worker metrics
```

### Components

| Path | Language | Role |
|------|----------|------|
| `neuralnet/worker.py` | Python | Training worker — forward/backward pass, sends gradients via gRPC |
| `neuralnet/network.py` | Python | `Layer` and `Network` classes; pure NumPy, no framework |
| `neuralnet/mnist.py` | Python | Reads MNIST binary format, normalizes to [0,1] |
| `neuralnet/data.py` | Python | Splits dataset into N equal shards by worker ID |
| `neuralnet/config.py` | Python | Loads `config.yaml` |
| `neuralnet/metrics.py` | Python | Prometheus gauges: accuracy, epoch time, epoch counter |
| `neuralnet/train.py` | Python | Single-node training baseline (no distributed overhead) |
| `server/main.go` | Go | Parameter server — gradient averaging, weight checkpointing |
| `server/config.go` | Go | Loads `config.yaml` for the server |
| `server/proto/` | Protobuf | gRPC service definition |
| `inference/server.py` | Python | Flask server, watches checkpoint, serves `/predict` |
| `inference/frontend/index.html` | HTML/JS | React digit-recognition canvas UI |

---

## The neural network

Architecture: `784 → 256 → 128 → 64 → 10` (fully connected, ReLU hidden layers, softmax output)

All implemented from scratch in NumPy — no PyTorch, no TensorFlow. The `Network` class does forward passes, backprop, and weight serialization. Weights are stored as a flat list for easy serialization over gRPC.

---

## Distributed training details

### Gradient compression

Workers compress gradients before sending by keeping only the top 10% by absolute magnitude (Top-K sparsification). This cuts communication volume by 10× with minimal accuracy loss.

```python
def compress_gradients(gradients, k_percent=0.1):
    k = max(1, int(len(gradients) * k_percent))
    top_k_indices = np.argsort(np.abs(gradients))[-k:]
    return top_k_indices.tolist(), gradients[top_k_indices].tolist()
```

### Synchronous gradient averaging

The parameter server waits for all N workers to submit gradients for a given batch before averaging and broadcasting the result back. Workers block on `cond.Wait()` until the last worker arrives and calls `cond.Broadcast()`.

### Fault tolerance

If a worker crashes mid-round, a 60-second timeout goroutine fires and the server proceeds with however many workers did respond, dynamically reducing `nWorkers` for the rest of that run:

```go
if s.receivedCount == 1 {
    go func() {
        time.Sleep(60 * time.Second)
        // if still waiting, proceed with fewer workers
        s.nWorkers = s.receivedCount
        s.computeAverage(totalSize)
        s.cond.Broadcast()
    }()
}
```

### Checkpointing and resumption

After each epoch, all workers send their full weights to the server. The server averages them (federated averaging), saves to `checkpoint.json`, and returns the averaged weights so all workers stay in sync. On restart, the server loads the checkpoint and initializes new workers from it — training picks up where it left off.

---

## Running it

### Prerequisites

- Docker + Docker Compose
- MNIST data files in `data/` (download below)

### Download MNIST

```bash
mkdir -p data && cd data
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

### Start distributed training

```bash
docker compose up
```

This starts all 8 containers: 4 workers, the parameter server, the inference server, Prometheus, and Grafana. Workers begin training immediately. The inference server comes up once the first checkpoint is written (after epoch 1, ~30s).

### Single-node training (baseline)

```bash
cd neuralnet
python train.py
```

Reaches ~97.6% accuracy by epoch 40 on the full dataset.

### View training metrics

Open Grafana at [http://localhost:3000](http://localhost:3000) (admin / admin). The pre-provisioned dashboard shows per-worker accuracy, epoch time, and epoch count over time from Prometheus.

### Draw a digit

Open [http://localhost:5000](http://localhost:5000) — or the live demo at [distributedneuralnet.vincents.systems](https://distributedneuralnet.vincents.systems) — draw a digit on the 28×28 canvas, hit Predict.

---

## Configuration

All hyperparameters live in `config.yaml`:

```yaml
training:
  n_workers: 4
  epochs: 20
  learning_rate: 0.01
  batch_size: 32
  architecture: [784, 256, 128, 64, 10]

server:
  port: 50051
  workers: 4
  checkpoint_path: /app/checkpoint/checkpoint.json
  checkpoint_interval: 5

data:
  train_images: /app/data/train-images-idx3-ubyte
  train_labels: /app/data/train-labels-idx1-ubyte
  test_images:  /app/data/t10k-images-idx3-ubyte
  test_labels:  /app/data/t10k-labels-idx1-ubyte
```

---

## Project structure

```
distributedneuralnet/
├── config.yaml
├── docker-compose.yml
├── Dockerfile.worker
├── Dockerfile.server
├── Dockerfile.inference
├── data/                        # MNIST binary files (not committed)
├── neuralnet/
│   ├── worker.py                # Distributed training worker
│   ├── network.py               # Neural network (pure NumPy)
│   ├── mnist.py                 # MNIST data loader
│   ├── data.py                  # Dataset sharding
│   ├── config.py                # Config loader
│   ├── metrics.py               # Prometheus metrics
│   └── train.py                 # Single-node baseline
├── server/
│   ├── main.go                  # Parameter server
│   ├── config.go                # Config loader
│   ├── go.mod
│   └── proto/
│       ├── neural.proto
│       ├── neural_pb2.py
│       └── neural_pb2_grpc.py
├── inference/
│   ├── server.py                # Flask inference server
│   └── frontend/
│       └── index.html           # React digit-recognition UI
└── monitoring/
    ├── prometheus.yml
    └── grafana/
        └── dashboards/
```

---

## Deployment

The inference frontend is exposed publicly via a Cloudflare tunnel pointing to `localhost:5000`. The tunnel runs as a Docker container alongside the rest of the stack.

```bash
cloudflared tunnel login
cloudflared tunnel create distributedneuralnet
# add tunnel config to docker-compose.yml
```