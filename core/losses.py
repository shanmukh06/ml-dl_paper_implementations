import numpy as np

class MSE:
    @staticmethod
    def forward(y_pred, y_true):
        return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    @staticmethod
    def backward(y_pred, y_true):
        """Returns the derivative of MSE w.r.t. the predictions."""
        return (y_pred - y_true) / y_true.shape[0]

class CategoricalCrossEntropy:
    @staticmethod
    def forward(probs, y_true):
        """
        y_true: Integer array of class indices (not one-hot encoded).
        """
        samples = len(y_true)
        epsilon = 1e-15
        probs_safe = np.clip(probs, epsilon, 1.0 - epsilon)
        return np.mean(-np.log(probs_safe[range(samples), y_true]))

    @staticmethod
    def backward(probs, y_true):
        """
        Computes the gradient of Categorical Cross Entropy combined with Softmax.
        Assumes the layer preceding this loss is a Softmax activation.
        """
        batch_size = len(y_true)
        delta = probs.copy()
        delta[range(batch_size), y_true] -= 1
        delta /= batch_size
        return delta

class Softmax:
    """Softmax is typically executed in tandem with CategoricalCrossEntropy."""
    @staticmethod
    def forward(z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=1, keepdims=True)