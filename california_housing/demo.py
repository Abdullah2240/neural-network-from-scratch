import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.datasets import fetch_california_housing
from nn import DenseLayer, Activation_ReLU, Activation_Linear

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WEIGHTS_PATH = RESULTS_DIR / "model_weights.pkl"

if not WEIGHTS_PATH.exists():
    print(f"No weights found at {WEIGHTS_PATH}")
    print("Run train_scratch.py first to train and save the model.")
    sys.exit(1)

with open(WEIGHTS_PATH, "rb") as f:
    weights = pickle.load(f)

dense1 = DenseLayer(8, 64)
dense1.weights = weights["dense1_weights"]
dense1.biases = weights["dense1_biases"]
activation1 = Activation_ReLU()

dense2 = DenseLayer(64, 32)
dense2.weights = weights["dense2_weights"]
dense2.biases = weights["dense2_biases"]
activation2 = Activation_ReLU()

dense3 = DenseLayer(32, 1)
dense3.weights = weights["dense3_weights"]
dense3.biases = weights["dense3_biases"]
activation3 = Activation_Linear()

scaler_mean = weights["scaler_mean"]
scaler_std = weights["scaler_std"]

data = fetch_california_housing()
X_raw, y = data.data, data.target
X_scaled = ((X_raw - scaler_mean) / scaler_std).astype(np.float32)


def predict_batch(X):
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    dense3.forward(activation2.outputs)
    activation3.forward(dense3.outputs)
    return activation3.outputs.flatten()


# precompute all predictions for scatter
all_preds = predict_batch(X_scaled)

# set up figure
fig = plt.figure(figsize=(12, 5), facecolor="#1e1e2e")
fig.canvas.manager.set_window_title("California Housing Demo")

gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 0.08],
                      hspace=0.4, wspace=0.35,
                      left=0.08, right=0.96, top=0.88, bottom=0.06)

ax_scatter = fig.add_subplot(gs[0, 0])
ax_info = fig.add_subplot(gs[0, 1])
ax_btn = fig.add_subplot(gs[1, :])

title_text = fig.suptitle("", fontsize=15, fontweight="bold", color="white")


def show_random(_event=None):
    idx = np.random.randint(len(X_raw))
    actual = y[idx]
    predicted = all_preds[idx]
    error = predicted - actual

    # scatter plot
    ax_scatter.clear()
    ax_scatter.scatter(y, all_preds, s=1, alpha=0.15, color="#6272a4")
    ax_scatter.scatter([actual], [predicted], s=100, color="#e74c3c",
                       zorder=5, edgecolors="white", linewidths=1.5)
    ax_scatter.plot([0, 5.5], [0, 5.5], "--", color="#44475a", linewidth=1)
    ax_scatter.set_xlabel("Actual ($100k)", color="#cdd6f4", fontsize=10)
    ax_scatter.set_ylabel("Predicted ($100k)", color="#cdd6f4", fontsize=10)
    ax_scatter.set_facecolor("#1e1e2e")
    ax_scatter.tick_params(colors="#cdd6f4", labelsize=9)
    for spine in ax_scatter.spines.values():
        spine.set_color("#44475a")

    # info panel
    ax_info.clear()
    ax_info.set_facecolor("#1e1e2e")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    line_y = 0.92
    for name, val in zip(FEATURE_NAMES, X_raw[idx]):
        ax_info.text(0.05, line_y, f"{name}:", color="#6272a4",
                     fontsize=10, fontfamily="monospace",
                     transform=ax_info.transAxes)
        ax_info.text(0.55, line_y, f"{val:.2f}", color="#cdd6f4",
                     fontsize=10, fontfamily="monospace",
                     transform=ax_info.transAxes)
        line_y -= 0.1

    line_y -= 0.05
    ax_info.text(0.05, line_y, f"Predicted: ${predicted * 100_000:,.0f}",
                 color="#cdd6f4", fontsize=11, fontweight="bold",
                 transform=ax_info.transAxes)
    line_y -= 0.1
    ax_info.text(0.05, line_y, f"Actual:    ${actual * 100_000:,.0f}",
                 color="#cdd6f4", fontsize=11, fontweight="bold",
                 transform=ax_info.transAxes)

    err_color = "#2ecc71" if abs(error) < 0.5 else "#f39c12" if abs(error) < 1.0 else "#e74c3c"
    title_text.set_text(f"Error: ${error * 100_000:+,.0f}")
    title_text.set_color(err_color)

    fig.canvas.draw_idle()


btn = Button(ax_btn, "Next Random Sample",
             color="#44475a", hovercolor="#6272a4")
btn.label.set_color("white")
btn.label.set_fontsize(12)
btn.on_clicked(show_random)

show_random()
plt.show()
