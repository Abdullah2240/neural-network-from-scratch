import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tensorflow.keras.datasets import fashion_mnist
from nn import DenseLayer, Activation_ReLU, SoftmaxWithCategoricalCrossentropyLoss

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WEIGHTS_PATH = RESULTS_DIR / "model_weights.pkl"

if not WEIGHTS_PATH.exists():
    print(f"No weights found at {WEIGHTS_PATH}")
    print("Run train_scratch.py first to train and save the model.")
    sys.exit(1)

with open(WEIGHTS_PATH, "rb") as f:
    weights = pickle.load(f)

dense1 = DenseLayer(784, 128)
dense1.weights = weights["dense1_weights"]
dense1.biases = weights["dense1_biases"]
activation1 = Activation_ReLU()

dense2 = DenseLayer(128, 64)
dense2.weights = weights["dense2_weights"]
dense2.biases = weights["dense2_biases"]
activation2 = Activation_ReLU()

dense3 = DenseLayer(64, 10)
dense3.weights = weights["dense3_weights"]
dense3.biases = weights["dense3_biases"]

loss_activation = SoftmaxWithCategoricalCrossentropyLoss()

(_, _), (X_test, y_test) = fashion_mnist.load_data()


def predict(image_flat):
    dense1.forward(image_flat.reshape(1, -1))
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    dense3.forward(activation2.outputs)
    loss_activation.forward(dense3.outputs, np.array([0]))
    return loss_activation.outputs[0]


# set up the figure
fig = plt.figure(figsize=(10, 5), facecolor="#1e1e2e")
fig.canvas.manager.set_window_title("Fashion MNIST Demo")

gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.5], height_ratios=[1, 0.08],
                      hspace=0.4, wspace=0.3,
                      left=0.06, right=0.96, top=0.88, bottom=0.06)

ax_img = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1])
ax_btn = fig.add_subplot(gs[1, :])

title_text = fig.suptitle("", fontsize=15, fontweight="bold", color="white")


def show_random(_event=None):
    idx = np.random.randint(len(X_test))
    image = X_test[idx]
    flat = image.astype(np.float32).reshape(-1) / 255.0
    probs = predict(flat)
    predicted = np.argmax(probs)
    actual = y_test[idx]
    is_correct = predicted == actual

    # image
    ax_img.clear()
    ax_img.imshow(image, cmap="gray", interpolation="nearest")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"Actual: {CLASS_NAMES[actual]}", fontsize=11,
                     color="white", pad=8)
    for spine in ax_img.spines.values():
        spine.set_color("#44475a")

    # bars
    ax_bar.clear()
    colors = []
    for i in range(10):
        if i == predicted and is_correct:
            colors.append("#2ecc71")
        elif i == predicted and not is_correct:
            colors.append("#e74c3c")
        elif i == actual and not is_correct:
            colors.append("#f39c12")
        else:
            colors.append("#44475a")

    bars = ax_bar.barh(CLASS_NAMES, probs, color=colors, height=0.7, edgecolor="none")

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        if prob > 0.03:
            ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.0%}", va="center", fontsize=9, color="#cdd6f4")

    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_facecolor("#1e1e2e")
    ax_bar.tick_params(colors="#cdd6f4", labelsize=10)
    ax_bar.set_xlabel("Confidence", color="#cdd6f4", fontsize=10)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    mark = "CORRECT" if is_correct else "WRONG"
    color = "#2ecc71" if is_correct else "#e74c3c"
    title_text.set_text(f"Predicted: {CLASS_NAMES[predicted]}  —  {mark}")
    title_text.set_color(color)

    fig.canvas.draw_idle()


btn = Button(ax_btn, "Next Random Image",
             color="#44475a", hovercolor="#6272a4")
btn.label.set_color("white")
btn.label.set_fontsize(12)
btn.on_clicked(show_random)

show_random()
plt.show()
