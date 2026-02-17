from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X, y = data.data, data.target

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle("California Housing Dataset", fontsize=14)

# 8 feature histograms
for i in range(8):
    ax = axes[i // 3][i % 3]
    ax.hist(X[:, i], bins=50, color="#6272a4", edgecolor="none", alpha=0.8)
    ax.set_title(data.feature_names[i], fontsize=10)
    ax.tick_params(labelsize=8)

# target distribution
ax = axes[2][2]
ax.hist(y, bins=50, color="#2ecc71", edgecolor="none", alpha=0.8)
ax.set_title("MedHouseVal ($100k)", fontsize=10)
ax.tick_params(labelsize=8)

plt.tight_layout()

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
out_path = ASSETS_DIR / "dataset_preview.png"
plt.savefig(out_path, dpi=150)
plt.show()

print(f"Saved to {out_path}")
