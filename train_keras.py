import json
from tensorflow import keras
from tensorflow.keras import layers

# same data, same preprocessing
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255.0

# same architecture as scratch: 784 -> 128 -> 64 -> 10
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,),
                 kernel_regularizer=keras.regularizers.l2(5e-4)),
    layers.Dropout(0.1),
    layers.Dense(64, activation="relu",
                 kernel_regularizer=keras.regularizers.l2(5e-4)),
    layers.Dropout(0.1),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

results = {
    "train_acc": [float(v) for v in history.history["accuracy"]],
    "train_loss": [float(v) for v in history.history["loss"]],
    "val_acc": [float(v) for v in history.history["val_accuracy"]],
    "val_loss": [float(v) for v in history.history["val_loss"]],
    "test_acc": float(test_acc),
    "test_loss": float(test_loss),
}

with open("results/keras_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to results/keras_results.json")
