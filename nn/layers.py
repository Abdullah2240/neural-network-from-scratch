import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularization_l1=0., bias_regularization_l1=0.,
                 weight_regularization_l2=0., bias_regularization_l2=0.):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularization_l1 = weight_regularization_l1
        self.bias_regularization_l1 = bias_regularization_l1
        self.weight_regularization_l2 = weight_regularization_l2
        self.bias_regularization_l2 = bias_regularization_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularization_l1 > 0:
            dwL1 = np.ones_like(self.weights)
            dwL1[self.weights < 0] = -1
            self.dweights += self.weight_regularization_l1 * dwL1

        if self.bias_regularization_l1 > 0:
            dbL1 = np.ones_like(self.biases)
            dbL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularization_l1 * dbL1

        if self.weight_regularization_l2 > 0:
            self.dweights += 2 * self.weight_regularization_l2 * self.weights

        if self.bias_regularization_l2 > 0:
            self.dbiases += 2 * self.bias_regularization_l2 * self.biases


class DropoutLayer:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.outputs = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
