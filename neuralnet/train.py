from mnist import load_mnist
from network import Network
import numpy as np
import time
from metrics import plot_training_curve
from data import split_dataset

EPOCHS = 10
LEARNING_RATE = 0.0001
ARCHITECTURE = [784, 128, 64, 10]
WORKER_ID = 0
N_WORKERS = 4

TRAIN_IMAGES_PATH = "data/train-images-idx3-ubyte.gz"
TRAIN_LABELS_PATH = "data/train-labels-idx1-ubyte.gz"
TEST_IMAGES_PATH = "data/t10k-images-idx3-ubyte.gz"
TEST_LABELS_PATH = "data/t10k-labels-idx1-ubyte.gz"

train_data = split_dataset(list(load_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)), N_WORKERS, WORKER_ID)
test_data = list(load_mnist(TEST_IMAGES_PATH, TEST_LABELS_PATH))

neural_network = Network(ARCHITECTURE)
weights = neural_network.get_weights()
print(len(weights))
neural_network.set_weights(weights)
weights2 = neural_network.get_weights()
print(weights == weights2)

accuracies = []
cumulative_times = []
total_time = 0

for epoch in range(EPOCHS):

	start_time = time.time()

	for inputs, label in train_data:
		inputs = np.array(inputs)
		neural_network.train(inputs, label, LEARNING_RATE)

	correct = 0
	for inputs, label in test_data:
		inputs = np.array(inputs)
		prediction = neural_network.forward(inputs)
		if np.argmax(prediction) == label:
			correct += 1

	elapsed_time = time.time() - start_time
	accuracy = correct/len(test_data)*100
	total_time += elapsed_time

	print(f"Epoch {epoch}: accuracy={accuracy:.2f}% time={elapsed_time:.1f}")

	accuracies.append(accuracy)
	cumulative_times.append(total_time)

plot_training_curve(cumulative_times, accuracies)