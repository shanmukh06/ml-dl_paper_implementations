# Implementation of "Adam: A Method for Stochastic Optimization"

This project is a from-scratch implementation of the 2014 Adam optimizer paper by Diederik P. Kingma and Jimmy Ba, built using only NumPy.

**Paper:** [Adam: A Method forochastic Optimization](https://arxiv.org/abs/1412.6980) (ICLR, 2015). A local copy is included as `paper.pdf`.

## Project Status: Complete

This implementation successfully replicates the Adam and AdaMax optimization algorithms. It demonstrates superior convergence properties on complex, non-convex surfaces (specifically the Beale function) compared to standard Stochastic Gradient Descent (SGD).



## Project Structure

* **`model.py`**: Contains the core optimization classes built from scratch:
    * `Adam.step()`: Implements **Algorithm 1**, maintaining moving averages of the gradient ($m_t$) and its square ($v_t$).
    * `AdaMax.step()`: Implements the $L^\infty$ norm variant described in **Section 7**.
    * **Bias Correction**: Implements the initialization bias correction terms $\hat{m}_t$ and $\hat{v}_t$ to prevent sluggish initial steps.
    * **Numerical Stability**: Handles epsilon ($\epsilon$) placement for stable division in deep valleys.

* **`experiments.ipynb`**: A comprehensive analysis notebook that:
    1.  Breaks down the mathematical derivation of **Section 3** (Bias Correction).
    2.  Implements the **Beale Function** benchmark, a notorious test for adaptive optimizers.
    3.  Visualizes the optimization trajectories of **Adam vs. SGD**.
    4.  Demonstrates robustness against exploding gradients using gradient clipping.

* **`train.py`**: A standalone script that:
    1.  Initializes a weight vector for a non-convex optimization task.
    2.  Uses the `Adam` class to minimize the objective function.
    3.  Logs loss and parameter coordinates to show convergence speed.

## How to Run

1.  Ensure you have the required libraries:
    ```bash
    pip install numpy matplotlib
    ```
2.  Run the experiment notebook:
    Open `experiments.ipynb` to see the comparative visualizations.
3.  Run the benchmark script:
    ```bash
    python train.py
    ```

You will observe how Adam's adaptive learning rate allows it to navigate flat plateaus and steep ravines where vanilla SGD typically oscillates or fails to converge without extreme hyperparameter tuning.

---
