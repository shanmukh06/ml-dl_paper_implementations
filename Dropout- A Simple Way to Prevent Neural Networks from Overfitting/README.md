# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

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