import json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / "scratch_results.json") as f:
    scratch = json.load(f)

with open(RESULTS_DIR / "keras_results.json") as f:
    keras = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("California Housing Training Curves", fontsize=14, fontweight="bold")

epochs_s = range(1, len(scratch["train_loss"]) + 1)
epochs_k = range(1, len(keras["train_loss"]) + 1)

# loss (MSE)
ax1.plot(epochs_s, scratch["train_loss"], "-o", color="#e74c3c", markersize=3, label="Scratch")
ax1.plot(epochs_k, keras["train_loss"], "-s", color="#3498db", markersize=3, label="Keras")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Training Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# MAE
ax2.plot(epochs_s, scratch["train_mae"], "-o", color="#e74c3c", markersize=3, label="Scratch")
ax2.plot(epochs_k, keras["train_mae"], "-s", color="#3498db", markersize=3, label="Keras")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MAE ($100k)")
ax2.set_title("Training MAE")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out_path = ASSETS_DIR / "training_curves.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved to {out_path}")
