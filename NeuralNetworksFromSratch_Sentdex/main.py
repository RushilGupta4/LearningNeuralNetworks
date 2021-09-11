import numpy as np
import time

np.random.seed(0)


# ################################################################################################### #
# ########################################## NNFS Functions  ######################################## #
# ################################################################################################### #

# Datasets - spiral_data
def spiral_data(samples, classes):
	x_coords = np.zeros((samples*classes, 2))
	y_coords = np.zeros(samples*classes, dtype="uint8")
	for class_number in range(classes):
		ix = range(samples*class_number, samples*(class_number+1))
		r = np.linspace(0.0, 1, samples)
		t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
		x_coords[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
		y_coords[ix] = class_number
	return x_coords, y_coords

# ################################################################################################### #
# ########################################### End Functions  ######################################## #
# ################################################################################################### #


# ################################################################################################### #
# ############################################## Classes  ########################################### #
# ################################################################################################### #

# Layer - Dense
class LayerDense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		self.output = None

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


# Activation - Relu
class ActivationReLU:
	def __init__(self):
		self.output = None

	def forward(self, inputs):
		self.output = np.maximum(0, inputs)


# Activation - Softmax
class ActivationSoftmax:
	def __init__(self):
		self.output = None

	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities


# Loss - Base
class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		return data_loss


# Loss - CategoricalCrossEntropy
class LossCategoricalCrossEntropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-8, 1-1e-8)

		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
		else:
			return

		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods


# ################################################################################################### #
# ############################################ End Classes  ######################################### #
# ################################################################################################### #


start = time.perf_counter()

X_train, y_train = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X_train)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = LossCategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y_train)

print("Loss: ", loss)

finish = time.perf_counter()

print(f"\nTook {round((finish - start), 2)}s To Finish")