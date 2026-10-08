from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import sys
import os
import threading
import time
from prometheus_client import Counter, start_http_server

sys.path.append('/app/neuralnet')
from network import Network

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

ARCHITECTURE = [784, 256, 128, 64, 10]
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', '/app/checkpoints/model.json')

PREDICTION_COUNT = Counter('inference_predictions_total', 'Total predictions made')
DIGIT_COUNT = Counter('inference_digit_total', 'Predictions per digit', ['digit'])

net = None
last_modified = 0
lock = threading.Lock()

def load_model():
	global net
	if not os.path.exists(CHECKPOINT_PATH):
		print(f"Checkpoint not found, waiting for training...", flush=True)
		return
	with open(CHECKPOINT_PATH) as f:
		data = json.load(f)
	new_net = Network(ARCHITECTURE)
	new_net.set_weights(data['weights'])
	with lock:
		net = new_net
	print(f"Model reloaded from {CHECKPOINT_PATH}", flush=True)

def watch_checkpoint():
	global last_modified
	while True:
		try:
			if os.path.exists(CHECKPOINT_PATH):
				mtime = os.path.getmtime(CHECKPOINT_PATH)
				if mtime > last_modified:
					last_modified = mtime
					load_model()
		except Exception as e:
			print(f"Watcher error: {e}", flush=True)
			time.sleep(10)

start_http_server(5001)
load_model()
threading.Thread(target=watch_checkpoint, daemon=True).start()

@app.route('/')
def index():
	return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	with lock:
		if net is None:
			return jsonify({'error': 'Model not ready, training in progress'}), 503
		pixels = np.array(request.json['pixels'])
		probs = net.forward(pixels)
	digit = int(np.argmax(probs))
	PREDICTION_COUNT.inc()
	DIGIT_COUNT.labels(digit=str(digit)).inc()
	return jsonify({
		'digit': digit,
		'probabilities': probs.tolist()
	})

@app.route('/health')
def health():
  	return jsonify({'status': 'ok'})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)