import numpy as np
from model import build_dropout_mlp
from core.losses import CategoricalCrossEntropy, Softmax
from core.optimizers import Adam


def generate_overfitting_scenario_data():
    """
    Generates a small synthetic dataset with high dimensionality to
    create a scenario where dropout regularization is clearly visible.
    """
    np.random.seed(42)
    X_train = np.random.randn(200, 100)
    y_train = np.random.randint(0, 2, size=200)

    X_val = np.random.randn(100, 100)
    y_val = np.random.randint(0, 2, size=100)

    return X_train, y_train, X_val, y_val


def compute_accuracy(model, X, y):
    """Computes categorical classification accuracy."""
    logits, _ = model.forward(X)
    probs = Softmax.forward(logits)
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions == y) * 100


def train():
    EPOCHS = 30
    BATCH_SIZE = 32
    INPUT_SIZE = 100
    HIDDEN_SIZES = [256, 256]
    NUM_CLASSES = 2
    LEARNING_RATE = 0.001
    DROP_PROB = 0.5

    X_train, y_train, X_val, y_val = generate_overfitting_scenario_data()

    model = build_dropout_mlp(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, drop_probability=DROP_PROB)
    loss_fn = CategoricalCrossEntropy()
    optimizer = Adam(model.get_parameters(), lr=LEARNING_RATE)

    print("Starting Training Loop with Dropout Regularization...")
    for epoch in range(EPOCHS):
        model.train()

        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0
        batches = 0

        for i in range(0, X_train.shape[0], BATCH_SIZE):
            X_batch = X_shuffled[i:i + BATCH_SIZE]
            y_batch = y_shuffled[i:i + BATCH_SIZE]

            logits, caches = model.forward(X_batch)
            probs = Softmax.forward(logits)

            loss = loss_fn.forward(probs, y_batch)
            epoch_loss += loss
            batches += 1

            delta = loss_fn.backward(probs, y_batch)
            model.backward(delta, caches)

            optimizer.step(model.get_parameters(), model.get_gradients())

        model.eval()
        train_acc = compute_accuracy(model, X_train, y_train)
        val_acc = compute_accuracy(model, X_val, y_val)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {epoch_loss / batches:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")


if __name__ == "__main__":
    train()