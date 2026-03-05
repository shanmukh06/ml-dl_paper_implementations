import numpy as np
class Adam:
    """
        Implementation of Adam: A Method for Stochastic Optimization.
    """
    def __init__(self, params_shape_list, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = [np.zeros(shape) for shape in params_shape_list]
        self.v = [np.zeros(shape) for shape in params_shape_list]

    def step(self, params, gradients):
        self.t += 1

        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i]**2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            params[i] -= self.lr*m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

class AdaMax(Adam):
    """
    AdaMax variant of Adam based on the infinity norm (Section 7 of the paper).
    """

    def step(self, params, gradients):
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = np.maximum(self.beta2 * self.v[i], np.abs(gradients[i]))

            m_hat = self.m[i] / bias_correction1

            params[i] -= self.lr * m_hat / (self.v[i]+self.epsilon)
        return params
