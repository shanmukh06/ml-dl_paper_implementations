# Deep Residual Learning for Image Recognition

This directory contains the mathematical verification of **Residual Connections (ResNet)** as formulated by He et al. (2015). 

## 🔬 The Problem: The Degradation Problem
While Batch Normalization solved the vanishing gradient problem (allowing deep networks to converge), researchers discovered a new anomaly: as network depth increases, accuracy gets saturated and then degrades rapidly. This is an optimization problem; solvers struggle to approximate the identity mapping using stacked non-linear layers.

## ⚙️ Mathematical Formulation

### Forward Pass (Residual Addition Node)
Instead of hoping stacked layers $\mathcal{F}(x)$ fit the desired underlying mapping $\mathcal{H}(x)$, we explicitly force the layers to fit a *residual* mapping. We introduce a skip connection bypassing the layers:

$$\mathcal{H}(x) = \mathcal{F}(x) + x$$

If the optimal mapping is an identity function, it is far easier for the optimizer to drive the weights of $\mathcal{F}(x)$ to zero than to learn an identity transformation using a stack of non-linear matrix multiplications.

### Analytical Backward Pass
The computational graph introduces an addition node at the convergence of the skip connection and the main transformation path. According to multivariate calculus, the gradient distributes identically across an addition node:

$$\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}} \cdot \frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}} \left( \frac{\partial \mathcal{F}(x)}{\partial x} + 1 \right)$$

The $+ 1$ term guarantees that the gradient propagates directly back to earlier layers regardless of the weight values inside $\mathcal{F}(x)$. This mathematically eliminates the degradation of gradient magnitudes.

## 📁 Implementation Details
* `model.py`: Constructs competing ultra-deep MLPs, toggling the `core.layers.ResidualBlock` container.
* `train.py`: Compares optimization efficiency, demonstrating that the ResNet variant consistently achieves lower training error than the Plain network of identical depth.