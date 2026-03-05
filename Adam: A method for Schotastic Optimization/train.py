import numpy as np
from model import Adam


def generate_data(n_samples=100):
    """Generates a simple 2D dataset for binary classification."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true, y_pred):
    """Binary Cross-Entropy Loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train():
    X, y = generate_data(200)
    n_features = X.shape[1]

    weights = np.zeros((n_features, 1))
    bias = np.zeros((1, 1))
    params = [weights, bias]

    optimizer = Adam(params_shape_list=[p.shape for p in params], lr=0.01)

    print("Starting Training with Adam Optimizer...")
    print("-" * 30)

    epochs = 1000
    for epoch in range(epochs + 1):
        z = np.dot(X, params[0]) + params[1]
        predictions = sigmoid(z)

        loss = compute_loss(y, predictions)

        m = y.shape[0]
        dz = predictions - y
        dw = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)
        grads = [dw, db]

        params = optimizer.step(params, grads)

        if epoch % 100 == 0:
            preds_class = (predictions > 0.5).astype(float)
            accuracy = np.mean(preds_class == y) * 100
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("-" * 30)
    print("Training Complete!")
    print(f"Final Weights:\n{params[0]}")
    print(f"Final Bias: {params[1]}")


if __name__ == "__main__":
    train()