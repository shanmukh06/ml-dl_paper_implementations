import numpy as np
from model import (initialize_network, forward_pass, backward_pass, update_weights, calculate_error, sigmoid, sigmoid_derivative)

#DEMONSTRATING FIG 1 EXAMPLE OF SYMMETRIC DETECTION

def create_symmetry_dataset():
    x=[]
    y=[]

    #Taking
    for i in range(2**6):
        binary_str = format(i, '06b')

        input_vector = np.array([float(bit) for bit in binary_str])

        is_symmetric = (binary_str == binary_str[::-1])

        label = np.array([1.0 if is_symmetric else 0.0])

        x.append(input_vector.reshape(1,6))
        y.append(label.reshape(1,1))

    return x, y
X_train, y_train = create_symmetry_dataset()

LEARNING_RATE = 0.1
MOMENTUM = 0.9
EPOCHS = 1500

layer_sizes = [6, 2, 1]

weights, biases = initialize_network(layer_sizes)

prev_weight_updates = [np.zeros_like(w) for w in weights]
prev_bias_updates = [np.zeros_like(b) for b in biases]

print("\n--- Starting Training ---")
print(f"Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}, Momentum: {MOMENTUM}")
epoch_errors = []
for epoch in range(EPOCHS):
    total_epoch_error = 0
    for x_case, y_case in zip(X_train, y_train):
        final_output, cache = forward_pass(x_case, weights, biases)
        total_epoch_error += calculate_error(final_output, y_case)
        weight_grads, bias_grads = backward_pass(final_output, y_case,cache, weights)
        (weights, biases,
         prev_weight_updates,
         prev_bias_updates) = update_weights(weights, biases,
                                             weight_grads, bias_grads,
                                             prev_weight_updates, prev_bias_updates,
                                             LEARNING_RATE, MOMENTUM)
        epoch_errors.append(total_epoch_error / len(X_train))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, "
                  f"Error: {epoch_errors[-1]:.6f}")

print("\n--- Testing Trained Network ---")

correct_predictions = 0

for i, (x_case, y_case) in enumerate(zip(X_train, y_train)):
    final_output, _ = forward_pass(x_case, weights, biases)
    prediction = 1.0 if final_output[0][0] > 0.5 else 0.0
    correct_label = y_case[0][0]

    if prediction == correct_label:
        correct_predictions += 1

    if i % 10 == 0:
        print(f"Input: {x_case.flatten()}, "
              f"Target: {correct_label}, "
              f"Output: {final_output[0][0]:.3f}, "
              f"Prediction: {prediction}")

accuracy = (correct_predictions / len(X_train)) * 100
print(f"\nTraining Complete.")
print(f"Final Accuracy: {correct_predictions} / {len(X_train)} "
      f"({accuracy:.2f}%)")

