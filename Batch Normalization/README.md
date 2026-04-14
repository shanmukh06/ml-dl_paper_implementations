# Batch Normalization: Accelerating Deep Network Training

This directory contains the mathematical verification of **Batch Normalization** as formulated by Ioffe & Szegedy (2015). 

## 🔬 The Problem: Internal Covariate Shift
In deep networks, the distribution of each layer's inputs changes during training as the parameters of the previous layers change. This requires lower learning rates and careful parameter initialization, and makes it notoriously difficult to train models with saturating nonlinearities (like Sigmoid or Tanh).

## ⚙️ Mathematical Formulation

### Forward Pass (Training)
During training, the layer computes the mean $\mu_B$ and variance $\sigma_B^2$ of the current mini-batch. It then normalizes the activations into $x_{hat}$ and applies learnable scale $\gamma$ and shift $\beta$ parameters:

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma\hat{x}_i + \beta$$

*Note: Simultaneously, the layer updates an Exponential Moving Average (EMA) of the mean and variance to be utilized during inference.*

### Analytical Backward Pass
Implementing the computational graph for the backward pass of Batch Normalization is a rigorous calculus exercise. The gradient with respect to the input $x$ relies on the chain rule passing through the variance and mean paths:

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$
$$\frac{\partial L}{\partial \sigma_B^2} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot \frac{-1}{2}(\sigma_B^2 + \epsilon)^{-3/2}$$
$$\frac{\partial L}{\partial \mu_B} = \left( \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_B^2 + \epsilon}} \right) + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{1}{m} \sum_{i=1}^{m} -2(x_i - \mu_B)$$
$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_B^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}$$

## 📁 Implementation Details
* `model.py`: Constructs competing deep MLPs using `core.layers.BatchNorm1D`.
* `train.py`: Handles stateful execution (`.train()` vs `.eval()`) to toggle between mini-batch statistics and running averages.