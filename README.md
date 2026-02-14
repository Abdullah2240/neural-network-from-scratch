# Neural Network from Scratch

A fully connected neural network implemented from scratch using only NumPy, trained on MNIST. I also trained the same architecture in Keras to compare how close my implementation gets.

## What this is

I built every component of a neural network by hand — dense layers, activations (ReLU, softmax), categorical cross-entropy loss, dropout, L1/L2 regularization, and the Adam optimizer. All with forward and backward passes derived from the math.

Then I trained it on MNIST digits and compared it against the exact same architecture in Keras to see how it stacks up.

## Architecture

Same architecture for both implementations:

```
Input (784) -> Dense(128) -> ReLU -> Dropout(0.1)
            -> Dense(64)  -> ReLU -> Dropout(0.1)
            -> Dense(10)  -> Softmax
```

Adam optimizer, L2 regularization (5e-4), batch size 128, 10 epochs.

## Results

| | Test Accuracy | Test Loss |
|---|---|---|
| From Scratch | — | — |
| Keras | — | — |

*(fill in after running)*

## Project structure

```
nn/                    the from-scratch library
  layers.py            dense layer, dropout
  activations.py       relu, softmax
  losses.py            cross-entropy loss, combined softmax+loss
  optimizers.py        adam optimizer
train_scratch.py       train on MNIST with my implementation
train_keras.py         train on MNIST with Keras (same arch)
notebooks/             jupyter notebooks with training + comparison
results/               saved metrics from training runs
```

## Running it

```bash
pip install -r requirements.txt
python train_scratch.py
python train_keras.py
```
