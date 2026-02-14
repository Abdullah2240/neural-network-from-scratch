
import numpy as np
import nnfs
from nnfs.datasets  import spiral_data
nnfs.init()


# Dense Layer
class DenseLayer:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularization_l1=0., bias_regularization_l1=0.,
                 weight_regularization_l2=0., bias_regularization_l2=0.):
        # initializes the weights
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

        # Changing weights and biases with respect to the regularization loss
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



class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0




class Activation_Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.outputs = probabilities        



class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):
        regularization_loss = 0

        # L1 on weights
        if layer.weight_regularization_l1 > 0:
            regularization_loss += layer.weight_regularization_l1 * np.sum(np.abs(layer.weights))
        # L1 on biases
        if layer.bias_regularization_l1 > 0:
            regularization_loss += layer.bias_regularization_l1 * np.sum(np.abs(layer.biases))
        # L2 on weights
        if layer.weight_regularization_l2 > 0:
            regularization_loss += layer.weight_regularization_l2 * np.sum(layer.weights * layer.weights)
        # L2 on biases
        if layer.bias_regularization_l2 > 0:
            regularization_loss += layer.bias_regularization_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_true*y_pred_clipped, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            # One Hot Encoding
            y_true = np.eye(labels)[y_true]
        self.dinputs = - y_true / dvalues
        # Normalization
        self.dinputs = self.dinputs / samples




# Softmax Classifier - combined with softax activation
# and cross-entropy loss for faster backward step

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs

        return self.loss.calculate(self.outputs, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            # One Hot Encoding
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples



class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        # current learing rate is the one with alpha
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.beta1 = beta1
        self.beta2 = beta2

    # Call once before updation
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. /( 1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * (layer.dweights ** 2)
        layer.bias_cache   = self.beta2 * layer.bias_cache   + (1 - self.beta2) * (layer.dbiases ** 2)

        layer.weight_momentums = (self.beta1 * layer.weight_momentums) + ((1 - self.beta1) * layer.dweights)                                                                         
        layer.bias_momentums = (self.beta1 * layer.bias_momentums) + ((1 - self.beta1) * layer.dbiases)

        weight_momentums_updated = layer.weight_momentums / ((1 - self.beta1**(self.iterations+1)))
        bias_momentums_updated = layer.bias_momentums / ((1 - self.beta1**(self.iterations+1)))

        weight_cache_updated = layer.weight_cache / ((1 - self.beta2**(self.iterations+1)))
        bias_cache_updated = layer.bias_cache / ((1 - self.beta2**(self.iterations+1)))

        layer.weights += -self.current_learning_rate * \
                        weight_momentums_updated / \
                        (np.sqrt(weight_cache_updated) + self.epsilon)


        layer.biases += -self.current_learning_rate * \
                        bias_momentums_updated / \
                        (np.sqrt(bias_cache_updated) + self.epsilon)

    # Call once after updation
    def post_update_params(self):
        self.iterations += 1



class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.outputs = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# Trying the whole thing with L1/L2 regularization and Dropout
X, y = spiral_data(classes=3, samples=1000)
# dense1 = DenseLayer(2, 64)
dense1 = DenseLayer(2, 64, weight_regularization_l2=5e-4, bias_regularization_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = DenseLayer(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dropout1.forward(activation1.outputs) # # change flow
    dense2.forward(dropout1.outputs)

    #Loss Calculation
    data_loss = loss_activation.forward(dense2.outputs, y)
    regularization_loss = (
        loss_activation.loss.regularization_loss(dense1) +
        loss_activation.loss.regularization_loss(dense2)       
    )

    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.outputs, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    acc = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + 
              f'acc: {acc:.3f} ' + 
              f'data_loss: {data_loss:.3f} ' +
              f'regularization_loss: {regularization_loss:.3f} ' +
              f'total_loss: {loss:.3f} ' +
              f'lr: {optimizer.current_learning_rate:.8f}')

    # Back Pass
    loss_activation.backward(loss_activation.outputs, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs) # change flow
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)


    # Optimization
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test, y_test = spiral_data(classes=3, samples=100)
dense1.forward(X_test)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
loss = loss_activation.forward(dense2.outputs, y_test)

predictions = np.argmax(loss_activation.outputs, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
acc = np.mean(predictions == y_test)
print(f"evaluation, acc: {acc:.3f}, loss: {loss:.3f}")




