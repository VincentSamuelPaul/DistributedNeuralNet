import random

class Neuron:
	def __init__(self, n: int):
		self.n = n
		self.weights = []
		for i in range(self.n):
			self.weights.append(random.uniform(-1, 1))
		self.bias = random.uniform(-1, 1)

		def forward(self, inputs):
			answer = 0
			for i in range(self.n):
				answer += self.weights[i] * inputs[i]
			answer += self.bias
			return max(0, answer)

	def train(self, inputs, lr, upstream):
		predicted = self.forward(inputs)
		ReLU = None
		if predicted > 0:
			ReLU = 1
		else:
			ReLU = 0
		input_gradients = []
		for i in range(self.n):
			input_gradients.append(upstream * ReLU * self.weights[i])
			gradient = upstream * ReLU * inputs[i]
			self.weights[i] -= lr * gradient
		self.bias -= lr * upstream * ReLU
		return input_gradients