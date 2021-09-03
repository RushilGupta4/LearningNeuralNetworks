import numpy as np

np.random.seed(0)

# ################################################################################################### #
# ############################################# Functions  ########################################## #
# ################################################################################################### #

# Datasets - spiral_data
def spiral_data(samples, classes):
	X = np.zeros((samples*classes, 2))
	y = np.zeros(samples*classes, dtype="uint8")
	for class_number in range(classes):
		ix = range(samples*class_number, samples*(class_number+1))
		r = np.linspace(0.0, 1, samples)
		t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
		X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
		y[ix] = class_number
	return X, y

# ################################################################################################### #
# ########################################### End Functions  ######################################## #
# ################################################################################################### #


# ################################################################################################### #
# ############################################## Classes  ########################################### #
# ################################################################################################### #

# Layer - Dense
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

# Activation - Relu
class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

# Activation - Softmax
class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

# ################################################################################################### #
# ############################################ End Classes  ######################################### #
# ################################################################################################### #


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output)