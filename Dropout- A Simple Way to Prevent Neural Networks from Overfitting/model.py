import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.layers import Sequential, Linear, Dropout
from core.activations import ReLU


def build_dropout_mlp(input_size, hidden_sizes, output_size, drop_probability=0.5):
    """
    Constructs a Deep Multi-Layer Perceptron regularized with Inverted Dropout.
    Matches the setup analyzed in Srivastava et al. (2014).
    """
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(layer_sizes) - 1):
        # Linear projection
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], init_type='xavier'))

        # Apply activation and dropout on hidden layers only
        if i < len(layer_sizes) - 2:
            layers.append(ReLU())
            layers.append(Dropout(drop_probability=drop_probability))

    return Sequential(*layers)