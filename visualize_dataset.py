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
plt.savefig("assets/dataset_preview.png", dpi=150)
plt.show()
print("Saved to assets/dataset_preview.png")
