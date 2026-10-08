# import random, math
import numpy as np

class Layer:
    def __init__(self, n: int, n_neurons: int, activation = True):
        self.weights = np.random.randn(n_neurons, n) * np.sqrt(2.0 / n)
        self.bias = np.zeros(n_neurons)
        self.activation = activation

    def forward(self, inputs):
        self.last_input = inputs
        if inputs.ndim == 1:
            z = np.dot(self.weights, inputs) + self.bias
        else:
            z = np.dot(inputs, self.weights.T) + self.bias
        if self.activation:
            self.last_output = np.maximum(0, z)
        else:
            self.last_output = z
        return self.last_output

    def train(self, upstream, lr):
      if self.activation:
          relu_prime = (self.last_output > 0).astype(float)
          delta = upstream * relu_prime
      else:
          delta = upstream
      update = lr * np.outer(delta, self.last_input)
      self.weights -= update
      self.bias -= lr * delta
      return np.dot(self.weights.T, delta)
    
    def compute_gradients(self, upstream: np.ndarray):
      if self.activation:
          relu_prime = (self.last_output > 0).astype(float)
          delta = upstream * relu_prime
      else:
          delta = upstream
      
      weight_grads = np.dot(delta.T, self.last_input) / self.last_input.shape[0]
      bias_grads = delta.mean(axis=0)
      input_upstream = np.dot(delta, self.weights)

      return weight_grads, bias_grads, input_upstream
  

class Network:
	def __init__(self, layer_size: list):
		self.layers = []
		for i in range(1, len(layer_size)):
			if i == len(layer_size) - 1:
				self.layers.append(Layer(layer_size[i-1], layer_size[i], activation=False))
			else:
				self.layers.append(Layer(layer_size[i-1], layer_size[i], activation=True))

	def softmax(self, x: np.ndarray) -> np.ndarray:
		if x.ndim == 1:
			e = np.exp(x - np.max(x))
			return e / e.sum()
		else:
			e = np.exp(x - np.max(x, axis=1, keepdims=True))
			return e / e.sum(axis=1, keepdims=True)

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		output = inputs
		for layer in self.layers:
			output = layer.forward(output)
		return self.softmax(output)
  
	def train(self, inputs, label, lr):
		prediction = self.forward(inputs)
		one_hot = np.zeros(10)
		one_hot[label] = 1
		upstream = (prediction - one_hot)
		for _, layer in enumerate(reversed(self.layers)):
			upstream = layer.train(upstream, lr)

	def get_weights(self) -> list:
		flat = []
		for layer in self.layers:
			flat.extend(layer.weights.flatten())
			flat.extend(layer.bias.flatten())
		return flat

	def set_weights(self, weights: list) -> None:
		idx = 0
		for layer in self.layers:
			w_size = layer.weights.size
			b_size = layer.bias.size
			layer.weights = np.array(weights[idx:idx+w_size]).reshape(layer.weights.shape)
			idx += w_size
			layer.bias = np.array(weights[idx:idx+b_size])
			idx += b_size

	def compute_gradients(self, inputs: np.array, labels: int) -> list:
		predicted = self.forward(inputs)
		one_hot = np.zeros((len(labels), 10))

		for i, label in enumerate(labels):
			one_hot[i, label] = 1

		upstream = predicted - one_hot

		layer_grads = []
		for layer in reversed(self.layers):
			weight_grads, bias_grads, upstream = layer.compute_gradients(upstream)
			layer_grads.append((weight_grads, bias_grads))

		flat_grads = []
		for weight_grads, bias_grads in reversed(layer_grads):
			flat_grads.extend(weight_grads.flatten())
			flat_grads.extend(bias_grads.flatten())

		return flat_grads
  
	def apply_gradients(self, gradients: list, lr: float) -> None:
		idx = 0
		for layer in self.layers:
			w_size = layer.weights.size
			b_size = layer.bias.size
			layer.weights -= lr * np.array(gradients[idx:idx+w_size]).reshape(layer.weights.shape)
			idx += w_size
			layer.bias -= lr * np.array(gradients[idx:idx+b_size])
			idx += b_size