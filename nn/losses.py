import numpy as np
from .activations import Activation_Softmax


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularization_l1 > 0:
            regularization_loss += layer.weight_regularization_l1 * np.sum(np.abs(layer.weights))

        if layer.bias_regularization_l1 > 0:
            regularization_loss += layer.bias_regularization_l1 * np.sum(np.abs(layer.biases))

        if layer.weight_regularization_l2 > 0:
            regularization_loss += layer.weight_regularization_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularization_l2 > 0:
            regularization_loss += layer.bias_regularization_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


class CategoricalCrossentropyLoss(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_true * y_pred_clipped, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class SoftmaxWithCategoricalCrossentropyLoss:
    # combined softmax + cross-entropy for faster backward step
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = CategoricalCrossentropyLoss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        return self.loss.calculate(self.outputs, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
