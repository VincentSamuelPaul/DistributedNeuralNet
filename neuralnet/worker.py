import grpc
import sys
import numpy as np
import time
import os
import neural_pb2
import neural_pb2_grpc
from mnist import load_mnist
from network import Network
from data import split_dataset
from config import load_config
from metrics import start_metrics_server, ACCURACY, EPOCH_TIME, EPOCH

cfg = load_config()

WORKER_ID = int(sys.argv[1])
N_WORKERS = cfg["training"]["n_workers"]
EPOCHS = cfg["training"]["epochs"]
LEARNING_RATE = cfg["training"]["learning_rate"]
BATCH_SIZE = cfg["training"]["batch_size"]
ARCHITECTURE = cfg["training"]["architecture"]
K_PERCENT = cfg["training"]["k_percent"]
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", f"localhost:{cfg['server']['port']}")
TRAIN_IMAGES_PATH = cfg["data"]["train_images"]
TRAIN_LABELS_PATH = cfg["data"]["train_labels"]
TEST_IMAGES_PATH = cfg["data"]["test_images"]
TEST_LABELS_PATH = cfg["data"]["test_labels"]

METRICS_PORT = 8000 + WORKER_ID
start_metrics_server(METRICS_PORT)


def compress_gradients(gradients: np.ndarray, k_percent: float = 0.1) -> tuple:
    k = max(1, int(len(gradients) * k_percent))
    top_k_indices = np.argsort(np.abs(gradients))[-k:]
    values = gradients[top_k_indices]
    return top_k_indices.astype(np.int32).tolist(), values.tolist()

def decompress_gradients(indices: list, values: list, total_size: int) -> np.ndarray:
    full = np.zeros(total_size)
    for idx, val in zip(indices, values):
        full[idx] = val
    return full

def send_gradients_with_retry(stub, request, max_retries=5):
    delay = 1
    for attempt in range(max_retries):
        try:
            return stub.SendGradients(request)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"gRPC failed, retrying in {delay}s: {e}", flush=True)
            time.sleep(delay)
            delay *= 2


train_data = split_dataset(list(load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)), N_WORKERS, WORKER_ID)
test_data = list(load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH))

channel = grpc.insecure_channel(SERVER_ADDRESS)
stub = neural_pb2_grpc.ParameterServerStub(channel)

neural_network = Network(ARCHITECTURE)
total_size = len(neural_network.get_weights())

if WORKER_ID == 0:
  initial_weights = neural_network.get_weights()
  stub.InitializeWeights(neural_pb2.WeightInit(values=initial_weights))
  print("Worker 0 initialized weights on server", flush=True)
else:
  while True:
    response = stub.GetInitialWeights(neural_pb2.Empty())
    if len(response.values) > 0:
        neural_network.set_weights(list(response.values))
        print(f"Worker {WORKER_ID} received initial weights", flush=True)
        break
    time.sleep(1)


for epoch in range(EPOCHS):
    start = time.time()
    
    for batch_start in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[batch_start:batch_start + BATCH_SIZE]
        input_batch = np.array([x for x, _ in batch])
        labels_batch = [y for _, y in batch]
        
        grads = neural_network.compute_gradients(input_batch, labels_batch)
        
        indices, values = compress_gradients(np.array(grads), k_percent=K_PERCENT)
        response = send_gradients_with_retry(stub, neural_pb2.GradientRequest(
            worker_id=WORKER_ID,
            indices=indices,
            values=values,
            total_size=total_size,
        ))
        averaged = decompress_gradients(
            list(response.indices),
            list(response.values),
            response.total_size
        )
        neural_network.apply_gradients(averaged.tolist(), LEARNING_RATE)

    elapsed = time.time() - start
    correct = sum(
        1 for inputs, label in test_data
        if np.argmax(neural_network.forward(np.array(inputs))) == label
    )
    print(f"Worker {WORKER_ID} Epoch {epoch}: accuracy={correct/len(test_data)*100:.2f}% time={elapsed:.1f}s", flush=True)

    ACCURACY.labels(worker_id=WORKER_ID).set(correct/len(test_data)*100)
    EPOCH_TIME.labels(worker_id=WORKER_ID).set(elapsed)
    EPOCH.labels(worker_id=WORKER_ID).inc()

    weights = neural_network.get_weights()
    response = stub.SendWeights(neural_pb2.WeightRequest(
        worker_id=WORKER_ID,
        weights=[float(w) for w in weights]
    ))
    neural_network.set_weights(list(response.values))