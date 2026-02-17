import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing

# Load California Housing (same as scratch)
data = fetch_california_housing()
X, y = data.data, data.target

# Same split as scratch
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Same standardization
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Same architecture: 8 -> 64 -> 32 -> 1
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(8,),
                 kernel_regularizer=keras.regularizers.l2(5e-4)),
    layers.Dropout(0.1),
    layers.Dense(32, activation="relu",
                 kernel_regularizer=keras.regularizers.l2(5e-4)),
    layers.Dropout(0.1),
    layers.Dense(1, activation="linear"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
predictions = model.predict(X_test, verbose=0)
ss_res = np.sum((y_test.reshape(-1, 1) - predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
test_r2 = 1 - (ss_res / ss_tot)

print(f"\nTest MSE: {test_mse:.4f}  MAE: {test_mae:.4f}  R²: {test_r2:.4f}")

results = {
    "train_loss": [float(v) for v in history.history["loss"]],
    "train_mae": [float(v) for v in history.history["mae"]],
    "val_loss": [float(v) for v in history.history["val_loss"]],
    "val_mae": [float(v) for v in history.history["val_mae"]],
    "test_mse": float(test_mse),
    "test_mae": float(test_mae),
    "test_r2": float(test_r2),
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "keras_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_DIR / 'keras_results.json'}")
