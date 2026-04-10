import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.layers import Sequential, Linear, BatchNorm1D
from core.activations import Sigmoid


def build_comparison_mlp(input_size, hidden_sizes, output_size, use_batchnorm=True):
    """
    Constructs a Deep Multi-Layer Perceptron to test Internal Covariate Shift.
    Intentionally uses 'standard' initialization and 'Sigmoid' activations to
    induce vanishing gradients in the unnormalized model.
    """
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(layer_sizes) - 1):
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], init_type='standard'))

        if i < len(layer_sizes) - 2:
            if use_batchnorm:
                layers.append(BatchNorm1D(layer_sizes[i + 1]))

            layers.append(Sigmoid())

    return Sequential(*layers)