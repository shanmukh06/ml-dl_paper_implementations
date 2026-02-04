import numpy as np

class LinearLayer:
    def __init__(self, n_in, n_out, init_type='xavier'):
        self.n_in = n_in
        self.n_out = n_out

        if init_type == 'xavier':
            limit = np.sqrt(6/(n_in+n_out))
            self.W = np.random.uniform(-limit, limit, (n_in, n_out))
        else:
            limit = 1/np.sqrt(n_in)
            self.W = np.random.uniform(-limit, limit, (n_in, n_out))

        self.b = np.zeros((1, n_out))

        self.last_input = None
        self.last_output = None
        self.dW = None
        self.dB = None

    def forward(self, x):
        self.last_input = x
        self.last_output = np.dot(x, self.W) + self.b
        return self.last_output

    def backward(self, delta_in):
        self.dW = np.dot(self.last_input.T, delta_in)/delta_in.shape[0]
        self.dB = np.mean(delta_in, axis=0, keepdims=True)
        delta_out = np.dot(delta_in, self.W.T)
        return delta_out


class SigmoidActivation:
    def __init__(self):
        self.cached_output = None

    def forward(self, x):
        # Edge Case Fix: Prevent np.exp overflow
        x_safe = np.clip(x, -500, 500)
        self.cached_output = 1 / (1 + np.exp(-x_safe))
        return self.cached_output

    def backward(self, delta_in):
        # O(1) derivative using cached state
        local_gradient = self.cached_output * (1.0 - self.cached_output)
        return delta_in * local_gradient


class TanhActivation:
    def __init__(self):
        self.cached_output = None

    def forward(self, x):
        self.cached_output = np.tanh(x)
        return self.cached_output

    def backward(self, delta_in):
        local_gradient = 1.0 - (self.cached_output ** 2)
        return delta_in * local_gradient

class DeepMLP:
    def __init__(self, layer_sizes, activation='tanh', init_type='xavier'):
        self.layers = []
        self.activations = [] # New list to hold stateful activation objects

        # Map string to class
        act_mapping = {
            'sigmoid': SigmoidActivation,
            'tanh': TanhActivation
        }
        ActClass = act_mapping[activation]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i+1], init_type))
            # Append an activation object for every hidden layer
            if i < len(layer_sizes) - 2:
                self.activations.append(ActClass())

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            # Apply activation for hidden layers
            if i < len(self.layers) - 1:
                out = self.activations[i].forward(out)
        return out

    def backward(self, probs, y_true):
        batch_size = len(y_true)
        delta = probs.copy()
        delta[range(batch_size), y_true] -= 1
        delta /= batch_size

        for i in reversed(range(len(self.layers))):
            delta = self.layers[i].backward(delta)
            # Pass delta through the activation backward method for hidden layers
            if i > 0:
                delta = self.activations[i-1].backward(delta)

class Loss:

    @staticmethod
    def softmax(z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps/np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def negative_log_likelihood(probs, true_label):
        samples = len(true_label)
        return np.mean(-np.log(probs[range(samples), true_label] + 1e-15))
