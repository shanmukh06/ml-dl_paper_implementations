import numpy as np

class Module:
    """
    Abstract base class for all neural network components.
    """
    def forward(self, x):
        """
        Executes the forward pass.
        Returns:
            out: The activation output.
            cache: The state required to compute the backward pass.
        """
        raise NotImplementedError

    def backward(self, delta_in, cache):
        """
        Executes the backward pass via the chain rule.
        Returns:
            delta_out: The gradient with respect to the input of this module.
        """
        raise NotImplementedError

    def get_parameters(self):
        """Returns a list of references to the trainable parameter arrays."""
        return []

    def get_gradients(self):
        """Returns a list of references to the gradient arrays."""
        return []