import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from model import DeepMLP, Loss

def prepare_data():
    """
    Loads and splits MNIST
    """
    print("Loading MNIST...")

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    X = X/255.0
    y=y.astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=10000, random_state=42, stratify=y_train_full)

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val, config):

    model = DeepMLP(layer_sizes=config['layer_sizes'], activation= config['activation'], init_type=config['init_type'])

    learning_rate = config['lr']
    batch_size = 10

    stats = {"activation_means": [], "activation_stds": [], "grad_variances": []}


    for epoch in range(config['epochs']):
        indices = np.random.permutation(len(X_train))
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            logits = model.forward(X_batch)
            probs = Loss.softmax(logits)

            model.backward(probs, y_batch)

            for layer in model.layers:
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.dB

        sample_logits = model.forward(X_val[:300])
        means = [np.mean(a) for  a in model.layer_a]
        stds = [np.std(a) for  a in model.layer_a]
        stats["activation_means"].append(means)
        stats["activation_stds"].append(stds)

        print(f"Epoch {epoch + 1} finished.")

    return model, stats

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    config_xavier = {
        'layer_sizes': [784, 1000, 1000, 1000, 1000, 10],
        'activation': 'tanh',
        'init_type': 'xavier',
        'lr': 0.01,
        'epochs': 50
    }

    model_n, stats_n = train_model(X_train, y_train, X_val, y_val, config_xavier)