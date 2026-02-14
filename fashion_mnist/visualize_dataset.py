from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

(X_train, y_train), _ = fashion_mnist.load_data()

fig, axes = plt.subplots(3, 6, figsize=(10, 5))
fig.suptitle("Fashion MNIST Samples", fontsize=14)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(CLASS_NAMES[y_train[i]], fontsize=9)
    ax.axis("off")

plt.tight_layout()
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
out_path = ASSETS_DIR / "dataset_preview.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved to {out_path}")
