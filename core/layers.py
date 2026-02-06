import numpy as np
from .module import Module


class Linear(Module):
    def __init__(self, n_in, n_out, init_type='xavier'):
        self.n_in = n_in
        self.n_out = n_out

        # Parameter Initialization
        if init_type == 'xavier':
            limit = np.sqrt(6 / (n_in + n_out))
            self.W = np.random.uniform(-limit, limit, (n_in, n_out))
        else:
            limit = 1 / np.sqrt(n_in)
            self.W = np.random.uniform(-limit, limit, (n_in, n_out))

        self.b = np.zeros((1, n_out))

        # Gradient Initialization
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        cache = x
        return out, cache

    def backward(self, delta_in, cache):
        x = cache

        # Compute gradients for parameters
        self.dW = np.dot(x.T, delta_in) / delta_in.shape[0]
        self.db = np.mean(delta_in, axis=0, keepdims=True)

        # Compute gradient for the next layer in the backward chain
        delta_out = np.dot(delta_in, self.W.T)
        return delta_out

    def get_parameters(self):
        return [self.W, self.b]

    def get_gradients(self):
        return [self.dW, self.db]


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        caches = []
        out = x
        for layer in self.layers:
            out, cache = layer.forward(out)
            caches.append(cache)
        return out, caches

    def backward(self, delta_in, caches):
        delta = delta_in
        for layer, cache in zip(reversed(self.layers), reversed(caches)):
            delta = layer.backward(delta, cache)
        return delta

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params

    def get_gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_gradients())
        return grads