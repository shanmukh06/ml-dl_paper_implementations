# Implementation of "Learning representations by back-propagating errors"

This project is a from-scratch implementation of the 1986 back-propagation paper by Rumelhart, Hinton, and Williams, built using only NumPy.

**Paper:** [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) (Nature, 1986). A local copy is included as `paper.pdf`.

## Project Status: Complete

This implementation successfully trains a neural network to solve the 6-bit symmetry detection task, achieving 100% accuracy on the full 64-case dataset.

## Project Structure

* **`model.py`**: Contains the core neural network functions, built from scratch:
    * `initialize_network()`: Sets up weights and biases.
    * `forward_pass()`: Implements Equations (1) and (2) for prediction.
    * `backward_pass()`: Implements Equations (4), (5), (6), and (7) for gradient calculation.
    * `update_weights()`: Implements Equation (9) (gradient descent with momentum).
    * `sigmoid()` and `sigmoid_derivative()`
    * `calculate_error()`

* **`train.py`**: The main executable script that:
    1.  Generates the 6-bit symmetry dataset.
    2.  Initializes a `[6, 2, 1]` network.
    3.  Trains the network for 1,500 epochs using the functions from `model.py`.
    4.  Tests the trained network and reports the final accuracy.

## How to Run

1.  Ensure you have NumPy installed:
    ```bash
    pip install numpy
    ```
2.  Run the `train.py` script:
    ```bash
    python train.py
    ```

You will see the training progress as the error decreases every 100 epochs, followed by a final test report.

