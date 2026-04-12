import numpy as np
from model import build_comparison_mlp
from core.losses import CategoricalCrossEntropy, Softmax
from core.optimizers import Adam


def generate_difficult_data():
    """Generates a synthetic dataset with high variance to test normalization."""
    np.random.seed(42)
    X_train = np.random.randn(500, 50) * 5 + 10
    y_train = np.random.randint(0, 3, size=500)

    X_val = np.random.randn(200, 50) * 5 + 10
    y_val = np.random.randint(0, 3, size=200)
    return X_train, y_train, X_val, y_val


def compute_accuracy(model, X, y):
    logits, _ = model.forward(X)
    probs = Softmax.forward(logits)
    return np.mean(np.argmax(probs, axis=1) == y) * 100


def train():
    EPOCHS = 25
    BATCH_SIZE = 64
    INPUT_SIZE = 50
    HIDDEN_SIZES = [128, 128, 128]
    NUM_CLASSES = 3
    LEARNING_RATE = 0.005

    X_train, y_train, X_val, y_val = generate_difficult_data()

    model_bn = build_comparison_mlp(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, use_batchnorm=True)
    loss_fn = CategoricalCrossEntropy()
    optimizer = Adam(model_bn.get_parameters(), lr=LEARNING_RATE)

    print("Training Deep Network WITH Batch Normalization...")
    for epoch in range(EPOCHS):
        model_bn.train()

        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled, y_shuffled = X_train[permutation], y_train[permutation]

        epoch_loss = 0
        batches = 0

        for i in range(0, X_train.shape[0], BATCH_SIZE):
            X_batch, y_batch = X_shuffled[i:i + BATCH_SIZE], y_shuffled[i:i + BATCH_SIZE]

            logits, caches = model_bn.forward(X_batch)
            probs = Softmax.forward(logits)

            loss = loss_fn.forward(probs, y_batch)
            epoch_loss += loss
            batches += 1

            delta = loss_fn.backward(probs, y_batch)
            model_bn.backward(delta, caches)
            optimizer.step(model_bn.get_parameters(), model_bn.get_gradients())

        model_bn.eval()
        val_acc = compute_accuracy(model_bn, X_val, y_val)
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {epoch_loss / batches:.4f} | Val Acc: {val_acc:.1f}%")


if __name__ == "__main__":
    train()