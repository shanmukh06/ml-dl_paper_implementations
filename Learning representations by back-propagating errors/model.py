import numpy as np

def initialize_network(layers):
    weights = []
    biases = []

    for i in range(1, len(layers)):
        input_dimensions = layers[i-1]
        output_dimensions = layers[i]

        w = np.random.uniform(-0.3, 0.3, size=(input_dimensions, output_dimensions))
        b = np.random.uniform(-0.3, 0.3, size=(1, output_dimensions))

        weights.append(w)
        biases.append(b)

    return weights, biases

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def forward_pass(input_vector, weights, biases):
    activations = [input_vector]
    inputs = []

    current_activation = input_vector

    for w, b in zip(weights, biases):
        total_inputs = np.dot(current_activation, w) + b
        current_activation = sigmoid(total_inputs)

        inputs.append(total_inputs)
        activations.append(current_activation)

    final_output = current_activation
    cache = (activations, inputs)
    return final_output, cache

def calculate_error(actual_output, desired_output):
    error = (1/2)* np.sum((actual_output-desired_output)**2)
    return error

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output*(1-sigmoid_output)

def backward_pass(final_output, desired_output, cache, weights):
    delta = final_output-desired_output
    delta = delta*sigmoid_derivative(final_output)

    activations, inputs = cache
    y_i = activations[-2]
    grad_w_output = np.dot(y_i.T, delta)

    grad_b_output = np.sum(delta, axis=0, keepdims=True)

    new_delta = np.dot(delta, weights[1].T)
    new_delta = new_delta*sigmoid_derivative(activations[1])

    y_k = activations[0]

    grad_w_hidden = np.dot(y_k.T, new_delta)

    grad_b_hidden = np.sum(new_delta, axis=0, keepdims=True)

    return (grad_w_hidden, grad_w_output), (grad_b_hidden, grad_b_output)

def update_weights(weights, biases, weight_gradients, bias_gradients,
                   prev_weight_updates, prev_bias_updates,
                   learning_rate, momentum):
    current_weight_updates = []
    current_bias_updates = []

    new_weights = []
    new_biases = []

    for w, b, grad_w, grad_b, prev_up_w, prev_up_b in zip(weights, biases, weight_gradients,
                                                          bias_gradients, prev_weight_updates, prev_bias_updates):
        delta_w = -learning_rate*grad_w+momentum*prev_up_w
        delta_b = -learning_rate * grad_b + momentum * prev_up_b

        new_w = w + delta_w
        new_b = b + delta_b

        new_weights.append(new_w)
        new_biases.append(new_b)

        current_bias_updates.append(delta_b)
        current_weight_updates.append(delta_w)

    return new_weights, new_biases, current_weight_updates, current_bias_updates