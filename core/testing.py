import numpy as np


def check_gradients(model, loss_fn, X, y, epsilon=1e-7, tolerance=1e-5):
    """
    Verifies analytical gradients against numerical approximations.
    """
    print("Initiating Gradient Check...")

    # 1. Compute Analytical Gradients (Framework's Backprop)
    predictions, caches = model.forward(X)
    loss = loss_fn.forward(predictions, y)
    delta = loss_fn.backward(predictions, y)
    model.backward(delta, caches)

    parameters = model.get_parameters()
    analytical_grads = model.get_gradients()

    # 2. Compute Numerical Gradients (Finite Differences)
    passed = True
    for param, analytic_grad in zip(parameters, analytical_grads):
        numerical_grad = np.zeros_like(param)

        # Iterate through every scalar inside the parameter matrix
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_value = param[idx]

            # f(x + epsilon)
            param[idx] = original_value + epsilon
            pred_plus, _ = model.forward(X)
            loss_plus = loss_fn.forward(pred_plus, y)

            # f(x - epsilon)
            param[idx] = original_value - epsilon
            pred_minus, _ = model.forward(X)
            loss_minus = loss_fn.forward(pred_minus, y)

            # Reset parameter
            param[idx] = original_value

            # Numerical derivative: (f(x+h) - f(x-h)) / 2h
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            it.iternext()

        # 3. Calculate Relative Error
        numerator = np.linalg.norm(analytic_grad - numerical_grad)
        denominator = np.linalg.norm(analytic_grad) + np.linalg.norm(numerical_grad)
        relative_error = numerator / (denominator + 1e-8)

        if relative_error > tolerance:
            print(f"Gradient Check FAILED! Error: {relative_error:.2e}")
            passed = False
        else:
            print(f"Gradient Check Passed. Error: {relative_error:.2e}")

    return passed