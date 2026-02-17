import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import numpy as np
from sklearn.datasets import fetch_california_housing
from nn import (
    DenseLayer, DropoutLayer, Activation_ReLU,
    Activation_Linear, MeanSquaredErrorLoss, Adam,
)

# Load California Housing
data = fetch_california_housing()
X, y = data.data, data.target  # (20640, 8), (20640,)

# Manual train/test split (80/20)
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Standardize features (fit on train only)
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = ((X_train - X_mean) / X_std).astype(np.float32)
X_test = ((X_test - X_mean) / X_std).astype(np.float32)

# Reshape targets to (N, 1)
y_train = y_train.reshape(-1, 1).astype(np.float32)
y_test = y_test.reshape(-1, 1).astype(np.float32)

# 8 -> 64 -> 32 -> 1
dense1 = DenseLayer(8, 64, weight_regularization_l2=5e-4, bias_regularization_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = DropoutLayer(0.1)

dense2 = DenseLayer(64, 32, weight_regularization_l2=5e-4, bias_regularization_l2=5e-4)
activation2 = Activation_ReLU()
dropout2 = DropoutLayer(0.1)

dense3 = DenseLayer(32, 1)
activation3 = Activation_Linear()

loss_function = MeanSquaredErrorLoss()
optimizer = Adam(learning_rate=0.001, decay=1e-4)

EPOCHS = 50
BATCH_SIZE = 128
history = {"train_loss": [], "train_mae": []}

for epoch in range(EPOCHS):
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0.0
    epoch_mae = 0.0
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
        activation3.forward(dense3.outputs)

        data_loss = loss_function.calculate(activation3.outputs, y_batch)
        regularization_loss = (
            loss_function.regularization_loss(dense1) +
            loss_function.regularization_loss(dense2) +
            loss_function.regularization_loss(dense3)
        )
        loss = data_loss + regularization_loss

        mae = np.mean(np.abs(activation3.outputs - y_batch))

        epoch_loss += loss
        epoch_mae += mae
        n_batches += 1

        # backward
        loss_function.backward(activation3.outputs, y_batch)
        activation3.backward(loss_function.dinputs)
        dense3.backward(activation3.dinputs)
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
    avg_mae = epoch_mae / n_batches
    history["train_loss"].append(float(avg_loss))
    history["train_mae"].append(float(avg_mae))

    print(f"epoch {epoch + 1}/{EPOCHS}  "
          f"loss: {avg_loss:.4f}  mae: {avg_mae:.4f}  "
          f"lr: {optimizer.current_learning_rate:.6f}")

# evaluation (no dropout)
dense1.forward(X_test)
activation1.forward(dense1.outputs)

dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)

dense3.forward(activation2.outputs)
activation3.forward(dense3.outputs)

predictions = activation3.outputs
test_mse = np.mean((predictions - y_test) ** 2)
test_mae = np.mean(np.abs(predictions - y_test))
ss_res = np.sum((y_test - predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
test_r2 = 1 - (ss_res / ss_tot)

print(f"\nTest MSE: {test_mse:.4f}  MAE: {test_mae:.4f}  R²: {test_r2:.4f}")

history["test_mse"] = float(test_mse)
history["test_mae"] = float(test_mae)
history["test_r2"] = float(test_r2)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "scratch_results.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"Results saved to {RESULTS_DIR / 'scratch_results.json'}")

# save model weights + scaler params
weights = {
    "dense1_weights": dense1.weights,
    "dense1_biases": dense1.biases,
    "dense2_weights": dense2.weights,
    "dense2_biases": dense2.biases,
    "dense3_weights": dense3.weights,
    "dense3_biases": dense3.biases,
    "scaler_mean": X_mean,
    "scaler_std": X_std,
}
weights_path = RESULTS_DIR / "model_weights.pkl"
with open(weights_path, "wb") as f:
    pickle.dump(weights, f)

print(f"Weights saved to {weights_path}")
