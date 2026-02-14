import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from nn import (
    DenseLayer, DropoutLayer, Activation_ReLU,
    SoftmaxWithCategoricalCrossentropyLoss, Adam,
)

# Load and flatten Fashion MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

# 784 -> 128 -> 64 -> 10
dense1 = DenseLayer(784, 128, weight_regularization_l2=5e-4, bias_regularization_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = DropoutLayer(0.1)

dense2 = DenseLayer(128, 64, weight_regularization_l2=5e-4, bias_regularization_l2=5e-4)
activation2 = Activation_ReLU()
dropout2 = DropoutLayer(0.1)

dense3 = DenseLayer(64, 10)

loss_activation = SoftmaxWithCategoricalCrossentropyLoss()
optimizer = Adam(learning_rate=0.001, decay=1e-4)

EPOCHS = 10
BATCH_SIZE = 128
history = {"train_acc": [], "train_loss": []}

for epoch in range(EPOCHS):
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0.0
    epoch_acc = 0.0
    n_batches = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        X_batch = X_shuffled[i:i + BATCH_SIZE]
        y_batch = y_shuffled[i:i + BATCH_SIZE]

        # forward
        dense1.forward(X_batch)
        activation1.forward(dense1.outputs)
        dropout1.forward(activation1.outputs)

        dense2.forward(dropout1.outputs)
        activation2.forward(dense2.outputs)
        dropout2.forward(activation2.outputs)

        dense3.forward(dropout2.outputs)

        data_loss = loss_activation.forward(dense3.outputs, y_batch)
        regularization_loss = (
            loss_activation.loss.regularization_loss(dense1) +
            loss_activation.loss.regularization_loss(dense2) +
            loss_activation.loss.regularization_loss(dense3)
        )
        loss = data_loss + regularization_loss

        predictions = np.argmax(loss_activation.outputs, axis=1)
        acc = np.mean(predictions == y_batch)

        epoch_loss += loss
        epoch_acc += acc
        n_batches += 1

        # backward
        loss_activation.backward(loss_activation.outputs, y_batch)
        dense3.backward(loss_activation.dinputs)
        dropout2.backward(dense3.dinputs)
        activation2.backward(dropout2.dinputs)
        dense2.backward(activation2.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        # optimize
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()

    avg_loss = epoch_loss / n_batches
    avg_acc = epoch_acc / n_batches
    history["train_acc"].append(float(avg_acc))
    history["train_loss"].append(float(avg_loss))

    print(f"epoch {epoch + 1}/{EPOCHS}  "
          f"acc: {avg_acc:.4f}  loss: {avg_loss:.4f}  "
          f"lr: {optimizer.current_learning_rate:.6f}")

# evaluation (no dropout)
dense1.forward(X_test)
activation1.forward(dense1.outputs)

dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)

dense3.forward(activation2.outputs)

test_loss = loss_activation.forward(dense3.outputs, y_test)
predictions = np.argmax(loss_activation.outputs, axis=1)
test_acc = np.mean(predictions == y_test)

print(f"\nTest accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

history["test_acc"] = float(test_acc)
history["test_loss"] = float(test_loss)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
with open(RESULTS_DIR / "scratch_results.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"Results saved to {RESULTS_DIR / 'scratch_results.json'}")

# save model weights
weights = {
    "dense1_weights": dense1.weights,
    "dense1_biases": dense1.biases,
    "dense2_weights": dense2.weights,
    "dense2_biases": dense2.biases,
    "dense3_weights": dense3.weights,
    "dense3_biases": dense3.biases,
}
weights_path = RESULTS_DIR / "model_weights.pkl"
with open(weights_path, "wb") as f:
    pickle.dump(weights, f)

print(f"Weights saved to {weights_path}")
