# Neural Network from Scratch

A fully connected neural network implemented from scratch using **only NumPy** — no PyTorch, no TensorFlow, no autograd. Every component (dense layers, activations, loss, dropout, regularization, optimizer) was hand-derived and implemented with raw math. Trained on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, with a Keras baseline using the exact same architecture for comparison.

## 📈 Results

| | Training Accuracy | Test Accuracy | Test Loss |
|---|---|---|---|
| **From Scratch (NumPy)** | 87.5% | **87.0%** | **0.360** |
| Keras Baseline | 88.2% | 87.0% | 0.429 |

The from-scratch implementation matches Keras in test accuracy and actually achieves a lower test loss.

## Dataset

![Fashion MNIST Samples](fashion_mnist/assets/dataset_preview.png)

10 classes of clothing items — T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots. Each image is 28x28 grayscale, flattened to 784 inputs.

## Architecture

Same architecture for both implementations:

| Layer | Units | Activation | Initialization | Regularization | Dropout |
|---|---|---|---|---|---|
| Input | 784 | — | — | — | — |
| Hidden 1 | 128 | ReLU | Random (0.01σ) | L2 (λ = 5e-4) | 0.1 |
| Hidden 2 | 64 | ReLU | Random (0.01σ) | L2 (λ = 5e-4) | 0.1 |
| Output | 10 | Softmax | — | — | — |

### Training Configuration

- **Loss Function:** Categorical Cross-Entropy + L2 regularization
- **Optimizer:** Adam (lr = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-7, decay = 1e-4)
- **Batch Size:** 128
- **Epochs:** 10
- **Data Shuffling:** Random permutation each epoch

## 📓 Step-by-Step Notebook

Want to see how I built every component from a single neuron to a full network? Check out the full learning journey on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kSnJVdCw2EiyyYwZw_wInFC0fGU8BGWB?usp=sharing)

## 🧠 What's Implemented from Scratch

Every forward and backward pass derived from the math:

- **Dense Layer** — matrix multiply forward, transposed gradient backward, with L1/L2 regularization gradients
- **ReLU Activation** — element-wise forward, masked gradient backward
- **Softmax + Cross-Entropy Loss** — combined for numerical stability, with the Jacobian shortcut in backprop
- **Dropout** — inverted dropout with scaled binary mask, gradient passthrough
- **Adam Optimizer** — momentum + RMSProp with bias correction on both moments

## 📂 Project Structure

```
nn/                              shared from-scratch library
  __init__.py                    public API
  layers.py                      dense layer, dropout
  activations.py                 relu, softmax
  losses.py                      cross-entropy loss, combined softmax+loss
  optimizers.py                  adam optimizer
fashion_mnist/                   Fashion MNIST classifier
  train_scratch.py               train with from-scratch implementation
  train_keras.py                 train with Keras (same arch)
  demo.py                        interactive inference with confidence bars
  visualize_dataset.py           preview grid of dataset samples
  results/                       saved metrics + model weights
  assets/                        images and visualizations
```

## 🛠️ Dependencies

- Python 3.x
- NumPy
- TensorFlow (only for loading the Fashion MNIST dataset)
- Matplotlib (for visualization)

## 👨‍💻 How to Run

```bash
git clone https://github.com/abdullah2240/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -r requirements.txt

python fashion_mnist/train_scratch.py      # train and save weights
python fashion_mnist/train_keras.py        # keras baseline for comparison
python fashion_mnist/demo.py               # interactive demo with confidence bars
python fashion_mnist/visualize_dataset.py  # preview the dataset
```

## License

MIT
