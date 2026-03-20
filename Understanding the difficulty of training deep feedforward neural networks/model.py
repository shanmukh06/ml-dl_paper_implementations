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

class Activations:

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        return 1.00 - np.tanh(x)**2

    @staticmethod
    def sigmoid(x):
        return 1 / ( 1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        s = 1/(1 + np.exp(-x))
        return s*(1-s)

    @staticmethod
    def softsign(x):
        return x/(1+np.abs(x))

    @staticmethod
    def softsign_prime(x):
        return 1/(1+np.abs(x))**2

class DeepMLP:
    def __init__(self, layer_sizes, activation='tanh', init_type='xavier'):
        """
        layer_sizes: list of int eg:[784, 1000, 1000, 1000, 1000, 10]:
        """

        self.layers = []
        self.activation_type = activation

        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i+1], init_type))

        self.act = getattr(Activations, activation)
        self.act_prime = getattr(Activations, f"{activation}_prime")

    def forward(self, x):
        self.layer_inputs = []
        self.layer_z = []
        self.layer_a = []

        out = x
        for i, layer in enumerate(self.layers):
            z = layer.forward(out)
            self.layer_z.append(z)

            if i < len(self.layers)-1:
                out = self.act(z)
                self.layer_a.append(out)
            else:
                out = z
        return out

    def backward(self, probs, y_true):
        batch_size = len(y_true)
        delta = probs.copy()
        delta[range(batch_size), y_true] -= 1
        delta /= batch_size

        for i in reversed(range(len(self.layers))):
            delta = self.layers[i].backward(delta)
            if i>0:
                delta = delta * self.act_prime(self.layer_z[i-1])


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
