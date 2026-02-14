import numpy as np


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]

    def step(self, params, gradients):
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i] ** 2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            # In-place parameter update
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdaMax(Adam):
    def step(self, params, gradients):
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = np.maximum(self.beta2 * self.v[i], np.abs(gradients[i]))

            m_hat = self.m[i] / bias_correction1

            # In-place parameter update
            params[i] -= self.lr * m_hat / (self.v[i] + self.epsilon)