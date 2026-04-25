import numpy as np
from model import build_plain_network, build_resnet
from core.losses import CategoricalCrossEntropy, Softmax
from core.optimizers import Adam


def generate_complex_data():
    """Generates a non-linear dataset requiring deep architectures to solve."""
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.sum(np.sin(X_train) + np.square(X_train), axis=1)
    y_train = np.digitize(y_train, bins=np.percentile(y_train, [25, 50, 75]))
    return X_train, y_train


def train():
    EPOCHS = 30
    BATCH_SIZE = 64
    INPUT_SIZE = 20
    HIDDEN_SIZE = 64
    NUM_BLOCKS = 5
    NUM_CLASSES = 4
    LEARNING_RATE = 0.005

    X_train, y_train = generate_complex_data()

    model_plain = build_plain_network(INPUT_SIZE, HIDDEN_SIZE, NUM_BLOCKS, NUM_CLASSES)
    model_resnet = build_resnet(INPUT_SIZE, HIDDEN_SIZE, NUM_BLOCKS, NUM_CLASSES)

    loss_fn = CategoricalCrossEntropy()
    opt_plain = Adam(model_plain.get_parameters(), lr=LEARNING_RATE)
    opt_resnet = Adam(model_resnet.get_parameters(), lr=LEARNING_RATE)

    print("Training 12-Layer Plain Network vs 12-Layer ResNet...")
    for epoch in range(EPOCHS):
        model_plain.train()
        model_resnet.train()

        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled, y_shuffled = X_train[permutation], y_train[permutation]

        loss_p, loss_r = 0, 0
        batches = 0

        for i in range(0, X_train.shape[0], BATCH_SIZE):
            X_batch, y_batch = X_shuffled[i:i + BATCH_SIZE], y_shuffled[i:i + BATCH_SIZE]
            batches += 1

            logits_p, caches_p = model_plain.forward(X_batch)
            loss_p += loss_fn.forward(Softmax.forward(logits_p), y_batch)
            model_plain.backward(loss_fn.backward(Softmax.forward(logits_p), y_batch), caches_p)
            opt_plain.step(model_plain.get_parameters(), model_plain.get_gradients())

            logits_r, caches_r = model_resnet.forward(X_batch)
            loss_r += loss_fn.forward(Softmax.forward(logits_r), y_batch)
            model_resnet.backward(loss_fn.backward(Softmax.forward(logits_r), y_batch), caches_r)
            opt_resnet.step(model_resnet.get_parameters(), model_resnet.get_gradients())

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | Plain Loss: {loss_p / batches:.4f} | ResNet Loss: {loss_r / batches:.4f}")


if __name__ == "__main__":
    train()