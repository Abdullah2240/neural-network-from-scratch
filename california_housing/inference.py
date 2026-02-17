import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from nn import DenseLayer, Activation_ReLU, Activation_Linear

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WEIGHTS_PATH = RESULTS_DIR / "model_weights.pkl"


def load_model(weights_path=WEIGHTS_PATH):
    with open(weights_path, "rb") as f:
        weights = pickle.load(f)

    dense1 = DenseLayer(8, 64)
    dense1.weights = weights["dense1_weights"]
    dense1.biases = weights["dense1_biases"]

    dense2 = DenseLayer(64, 32)
    dense2.weights = weights["dense2_weights"]
    dense2.biases = weights["dense2_biases"]

    dense3 = DenseLayer(32, 1)
    dense3.weights = weights["dense3_weights"]
    dense3.biases = weights["dense3_biases"]

    activation1 = Activation_ReLU()
    activation2 = Activation_ReLU()
    activation3 = Activation_Linear()

    scaler_mean = weights["scaler_mean"]
    scaler_std = weights["scaler_std"]

    return (dense1, activation1, dense2, activation2, dense3, activation3,
            scaler_mean, scaler_std)


def predict(features, model=None):
    """Predict median house values for one or more samples.

    Args:
        features: numpy array of shape (8,) or (N, 8), raw feature values.
        model: tuple from load_model(). Loads default weights if None.

    Returns:
        predictions: numpy array of shape (N, 1), values in $100k units.
    """
    if model is None:
        model = load_model()

    dense1, activation1, dense2, activation2, dense3, activation3, \
        scaler_mean, scaler_std = model

    features = np.atleast_2d(features).astype(np.float32)
    features = (features - scaler_mean) / scaler_std

    dense1.forward(features)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    dense3.forward(activation2.outputs)
    activation3.forward(dense3.outputs)

    return activation3.outputs


def predict_price(features, model=None):
    """Return predicted median house price in dollars."""
    raw = predict(features, model)
    return raw * 100_000


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X, y = data.data, data.target

    model = load_model()

    # single prediction
    idx = np.random.randint(len(X))
    pred = predict(X[idx], model)
    print(f"Single prediction: ${pred[0, 0] * 100_000:,.0f} "
          f"(actual: ${y[idx] * 100_000:,.0f})")

    # batch prediction
    preds = predict(X[:100], model)
    mae = np.mean(np.abs(preds.flatten() - y[:100]))
    print(f"Batch MAE (first 100): {mae:.4f} ($100k units)")
