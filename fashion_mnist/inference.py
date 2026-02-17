import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from nn import DenseLayer, Activation_ReLU, SoftmaxWithCategoricalCrossentropyLoss

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WEIGHTS_PATH = RESULTS_DIR / "model_weights.pkl"


def load_model(weights_path=WEIGHTS_PATH):
    with open(weights_path, "rb") as f:
        weights = pickle.load(f)

    dense1 = DenseLayer(784, 128)
    dense1.weights = weights["dense1_weights"]
    dense1.biases = weights["dense1_biases"]

    dense2 = DenseLayer(128, 64)
    dense2.weights = weights["dense2_weights"]
    dense2.biases = weights["dense2_biases"]

    dense3 = DenseLayer(64, 10)
    dense3.weights = weights["dense3_weights"]
    dense3.biases = weights["dense3_biases"]

    activation1 = Activation_ReLU()
    activation2 = Activation_ReLU()
    loss_activation = SoftmaxWithCategoricalCrossentropyLoss()

    return dense1, activation1, dense2, activation2, dense3, loss_activation


def predict(images, model=None):
    """Predict class probabilities for one or more images.

    Args:
        images: numpy array of shape (784,) or (N, 784), values 0-255 or 0-1.
        model: tuple from load_model(). Loads default weights if None.

    Returns:
        probs: numpy array of shape (N, 10) with class probabilities.
    """
    if model is None:
        model = load_model()

    dense1, activation1, dense2, activation2, dense3, loss_activation = model

    images = np.atleast_2d(images).astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0

    dense1.forward(images)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    dense3.forward(activation2.outputs)
    loss_activation.forward(dense3.outputs, np.zeros(len(images), dtype=int))

    return loss_activation.outputs


def classify(images, model=None):
    """Return predicted class names for one or more images."""
    probs = predict(images, model)
    indices = np.argmax(probs, axis=1)
    return [CLASS_NAMES[i] for i in indices]


if __name__ == "__main__":
    from tensorflow.keras.datasets import fashion_mnist

    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = load_model()

    # single prediction
    idx = np.random.randint(len(X_test))
    label = classify(X_test_flat[idx], model)
    print(f"Single prediction: {label[0]} (actual: {CLASS_NAMES[y_test[idx]]})")

    # batch prediction
    probs = predict(X_test_flat[:100], model)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_test[:100])
    print(f"Batch accuracy (first 100): {acc:.0%}")
