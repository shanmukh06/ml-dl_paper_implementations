# Implementation of "Learning representations by back-propagating errors"

This project is a from-scratch implementation of the 1986 back-propagation paper by Rumelhart, Hinton, and Williams, using only NumPy.

**Paper:** [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) (Nature, 1986). A local copy is included as `paper.pdf`.

## Project Goal

The goal is to implement the core mechanics of the back-propagation algorithm as described in the paper, including the forward pass, backward pass (gradient calculation), and weight update rule.

## Current Status

This is an initial commit with the core propagation functions implemented.

### Completed:

* **`model.py`** (Functional approach)
    * `initialize_network()`: Sets up weights and biases with small random values.
    * `sigmoid()`: Implements Equation (2).
    * `sigmoid_derivative()`: Efficiently calculates $dy/dx$ using the sigmoid's output.
    * `forward_pass()`: Implements Equations (1) and (2) to get a network output and a cache of intermediate values.
    * `backward_pass()`: Implements Equations (4), (5), (6), and (7) to calculate the error gradients for all weights and biases.

### To-Do:

* Implement the `update_weights()` function (Equation 9) to apply the calculated gradients with momentum.
* Create `train.py` to loop over a dataset (e.g., the symmetry problem) and train the network.
* Create `experiment.ipynb` to run experiments and visualize the results (like the loss curve).

## Usage

At this stage, the model can be imported in Python to perform a single forward and backward pass, but no training loop exists yet.