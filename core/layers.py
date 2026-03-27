import numpy as np
from .module import Module


class Linear(Module):
    def __init__(self, n_in, n_out, init_type='xavier'):
        super().__init__()
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
        super().__init__()
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


class Dropout(Module):
    def __init__(self, drop_probability=0.5):
        super().__init__()
        self.p = drop_probability

    def forward(self, x):
        if self.training:
            # Generate a binary mask where probability of 1 is (1 - p)
            mask = np.random.binomial(1, 1 - self.p, size=x.shape)

            # Apply inverted dropout scaling to maintain expectation
            mask = mask / (1.0 - self.p)

            out = x * mask
            cache = mask
            return out, cache
        else:
            # Identity pass during evaluation
            return x, None

    def backward(self, delta_in, cache):
        mask = cache
        if mask is None:
            return delta_in
        # The gradient is 0 for dropped units and scaled by 1/(1-p) for active units
        return delta_in * mask


class BatchNorm1D(Module):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable affine parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        # Inference state parameters (Exponential Moving Averages)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Gradient buffers
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x):
        if self.training:
            # Compute mini-batch statistics
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            # Update moving averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # Normalize
            x_hat = (x - mean) / np.sqrt(var + self.epsilon)

            # Scale and shift
            out = self.gamma * x_hat + self.beta

            cache = (x, x_hat, mean, var)
            return out, cache
        else:
            # Execute utilizing moving averages
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_hat + self.beta
            return out, None

    def backward(self, delta_in, cache):
        x, x_hat, mean, var = cache
        batch_size = x.shape[0]

        # Gradients w.r.t learnable parameters
        self.dgamma = np.sum(delta_in * x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(delta_in, axis=0, keepdims=True)

        # Gradient w.r.t normalized input
        dx_hat = delta_in * self.gamma

        # Gradient w.r.t variance
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + self.epsilon) ** (-1.5), axis=0, keepdims=True)

        # Gradient w.r.t mean
        dmean = np.sum(dx_hat * -1.0 / np.sqrt(var + self.epsilon), axis=0, keepdims=True)
        dmean += dvar * np.mean(-2.0 * (x - mean), axis=0, keepdims=True)

        # Gradient w.r.t input x
        dx = (dx_hat / np.sqrt(var + self.epsilon)) + (dvar * 2.0 * (x - mean) / batch_size) + (dmean / batch_size)

        return dx

    def get_parameters(self):
        return [self.gamma, self.beta]

    def get_gradients(self):
        return [self.dgamma, self.dbeta]


class ResidualBlock(Module):
    def __init__(self, main_path_module):
        """
        main_path_module: Typically a Sequential container consisting of
        Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm.
        """
        super().__init__()
        self.main_path = main_path_module

    def forward(self, x):
        # Execute the main transformation path
        main_out, main_cache = self.main_path.forward(x)

        # Identity skip connection (Addition Node)
        out = main_out + x

        cache = (main_cache, x)
        return out, cache

    def backward(self, delta_in, cache):
        main_cache, x = cache

        # Gradient routes through the main transformation block
        delta_main = self.main_path.backward(delta_in, main_cache)

        # Gradient routes identically through the skip connection (d(x)/dx = 1)
        delta_skip = delta_in

        # Sum the gradients at the fork convergence point
        dx = delta_main + delta_skip
        return dx

    def get_parameters(self):
        return self.main_path.get_parameters()

    def get_gradients(self):
        return self.main_path.get_gradients()

    def train(self):
        super().train()
        if hasattr(self.main_path, 'train'): self.main_path.train()

    def eval(self):
        super().eval()
        if hasattr(self.main_path, 'eval'): self.main_path.eval()