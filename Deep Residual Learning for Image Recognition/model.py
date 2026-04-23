import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.layers import Sequential, Linear, BatchNorm1D, ResidualBlock
from core.activations import ReLU


def build_plain_network(input_size, hidden_size, num_blocks, output_size):
    """
    Constructs a Deep 'Plain' Network (No skip connections).
    Even with BatchNorm, very deep plain networks suffer from optimization degradation.
    """
    layers = [Linear(input_size, hidden_size), BatchNorm1D(hidden_size), ReLU()]

    for _ in range(num_blocks):
        layers.extend([
            Linear(hidden_size, hidden_size), BatchNorm1D(hidden_size), ReLU(),
            Linear(hidden_size, hidden_size), BatchNorm1D(hidden_size), ReLU()
        ])

    layers.append(Linear(hidden_size, output_size))
    return Sequential(*layers)


def build_resnet(input_size, hidden_size, num_blocks, output_size):
    """
    Constructs a Deep Residual Network.
    Wraps consecutive layers in a ResidualBlock to allow gradients to flow
    unhindered through the identity mapping.
    """
    layers = [Linear(input_size, hidden_size), BatchNorm1D(hidden_size), ReLU()]

    for _ in range(num_blocks):
        block = Sequential(
            Linear(hidden_size, hidden_size), BatchNorm1D(hidden_size), ReLU(),
            Linear(hidden_size, hidden_size), BatchNorm1D(hidden_size)
        )
        layers.append(ResidualBlock(block))
        layers.append(ReLU())  # Final activation is applied AFTER the addition

    layers.append(Linear(hidden_size, output_size))
    return Sequential(*layers)