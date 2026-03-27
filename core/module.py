import numpy as np

class Module:
    """
    Abstract base class for all neural network components.
    Manages execution state (training vs evaluation) for stochastic layers.
    """
    def __init__(self):
        self.training = True

    def train(self):
        """Sets the module and all sub-modules to training mode."""
        self.training = True
        # Recursively apply to all sub-layers in container modules (like Sequential)
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if hasattr(layer, 'train'):
                    layer.train()

    def eval(self):
        """Sets the module and all sub-modules to evaluation mode."""
        self.training = False
        # Recursively apply to all sub-layers in container modules (like Sequential)
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if hasattr(layer, 'eval'):
                    layer.eval()

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