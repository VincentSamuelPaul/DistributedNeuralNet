from prometheus_client import Gauge, Counter, start_http_server

ACCURACY = Gauge('worker_accuracy', 'Training accuracy', ['worker_id'])
EPOCH_TIME = Gauge('worker_epoch_time', 'Time per epoch in seconds', ['worker_id'])
EPOCH = Counter('worker_epochs_total', 'Total epochs completed', ['worker_id'])
GRADIENT_SYNC_TIME = Gauge('worker_gradient_sync_time', 'Time for gradient sync', ['worker_id'])

def start_metrics_server(port: int):
    start_http_server(port)