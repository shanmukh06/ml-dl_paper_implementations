import numpy as np


class Tensor:
    """
    A Tensor that tracks its computation history to dynamically compute gradients
    via the chain rule (Directed Acyclic Graph).
    """

    def __init__(self, data, requires_grad=False, parents=None, op_name=""):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad

        # Initialize gradient buffer
        self.grad = np.zeros_like(self.data, dtype=float) if requires_grad else None

        # DAG Configuration: stores (parent_tensor, local_derivative_function)
        self.parents = parents or []
        self.op_name = op_name

    def zero_grad(self):
        """Clears the accumulated gradient."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=float)

    def backward(self, grad_in=None):
        """
        Executes the chain rule dynamically through the topological graph.
        """
        if not self.requires_grad:
            return

        # Seed the gradient at the end of the graph (usually a scalar loss of 1.0)
        if grad_in is None:
            grad_in = np.ones_like(self.data, dtype=float)

        self.grad += grad_in

        # Propagate gradients backward to all parents
        for parent, local_grad_fn in self.parents:
            if parent.requires_grad:
                # Chain Rule: dL/d_parent = dL/d_current * d_current/d_parent
                parent_grad = local_grad_fn(self.grad)
                parent.backward(parent_grad)

    # ==========================================
    # Computational Graph Operations (Operators)
    # ==========================================

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data

        # Derivative of A + B w.r.t A is 1. w.r.t B is 1.
        def grad_fn_self(grad): return grad

        def grad_fn_other(grad): return grad

        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(out_data, requires_grad,
                      parents=[(self, grad_fn_self), (other, grad_fn_other)],
                      op_name="Add")

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data

        # Derivative of matrix multiplication (A @ B)
        # dL/dA = dL/dOut @ B.T
        # dL/dB = A.T @ dL/dOut
        def grad_fn_self(grad): return grad @ other.data.T

        def grad_fn_other(grad): return self.data.T @ grad

        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(out_data, requires_grad,
                      parents=[(self, grad_fn_self), (other, grad_fn_other)],
                      op_name="MatMul")

    def relu(self):
        out_data = np.maximum(0, self.data)

        # Derivative of ReLU is 1 if x > 0 else 0# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

This directory contains the verification implementation of the **Inverted Dropout** regularizer as formulated by Srivastava et al. (2014). The execution utilizes the centralized `core/` framework to manage model states across execution phases.

## 🔬 Mathematical Formulation

### Forward Pass (Training)
Standard dropout scales the weights downward during inference to match expected activation levels. This implementation leverages **Inverted Dropout**, which instead modifies the forward activations during training. This removes the necessity of scaling operations during evaluation.

Given an input activation matrix $A$, a binary mask matrix $M$ is generated using a Bernoulli distribution with a retention probability $1 - p$:

$$M_{ij} \sim \text{Bernoulli}(1 - p)$$

The masked activations are scaled by the factor $\frac{1}{1 - p}$ to keep the expected value of the neuron inputs constant:

$$\hat{A} = \frac{A \odot M}{1 - p}$$

### Forward Pass (Evaluation)
During evaluation (`model.eval()`), the stochastic masking process is disabled entirely. The input is passed through unmodified to serve as a deterministic estimator:

$$\hat{A} = A$$

### Backward Pass
Since the operation is element-wise matrix multiplication with a constant scaling coefficient, the derivative with respect to the input activations propagates directly back through the exact same coordinate positions that were unmasked during the forward pass:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \hat{A}} \odot \frac{M}{1 - p}$$

## 📁 File Breakdown

* `model.py`: Assembles an Object-Oriented multi-layer perceptron topology integrating `core.layers.Linear`, `core.activations.ReLU`, and `core.layers.Dropout` blocks.
* `train.py`: Sets up a high-dimensional synthetic classification pipeline to isolate, test, and observe the convergence performance of regularized modules against validation splits.
        def grad_fn_self(grad):
            return grad * (self.data > 0).astype(float)

        return Tensor(out_data, self.requires_grad,
                      parents=[(self, grad_fn_self)],
                      op_name="ReLU")