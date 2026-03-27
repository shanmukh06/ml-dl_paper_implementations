import numpy as np
from .module import Module

class Sigmoid(Module):
    def forward(self, x):
        x_safe = np.clip(x, -500, 500)
        out = 1 / (1 + np.exp(-x_safe))
        cache = out
        return out, cache

    def backward(self, delta_in, cache):
        out = cache
        local_gradient = out * (1.0 - out)
        return delta_in * local_gradient

class Tanh(Module):
    def forward(self, x):
        out = np.tanh(x)
        cache = out
        return out, cache

    def backward(self, delta_in, cache):
        out = cache
        local_gradient = 1.0 - (out ** 2)
        return delta_in * local_gradient

class Softsign(Module):
    def forward(self, x):
        out = x / (1 + np.abs(x))
        cache = x
        return out, cache

    def backward(self, delta_in, cache):
        x = cache
        local_gradient = 1 / (1 + np.abs(x)) ** 2
        return delta_in * local_gradient

class ReLU(Module):
    """
    Rectified Linear Unit.
    Crucial for deep networks (like ResNet) to prevent vanishing gradients.
    """
    def forward(self, x):
        out = np.maximum(0, x)
        cache = x  # Cache the pre-activation input for the backward pass
        return out, cache

    def backward(self, delta_in, cache):
        x = cache
        # The local gradient is 1 where x > 0, and 0 where x <= 0
        local_gradient = (x > 0).astype(float)
        return delta_in * local_gradient