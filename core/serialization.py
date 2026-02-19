import numpy as np
import os


def save_weights(model, filepath):
    """
    Extracts all parameter arrays from the computational graph and
    serializes them to disk in a compressed .npz format.
    """
    params = model.get_parameters()

    # Create a dictionary mapping generic names to parameter arrays
    state_dict = {f"param_{i}": p for i, p in enumerate(params)}

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    # Save uncompressed arrays into a single zip file
    np.savez(filepath, **state_dict)
    print(f"Model weights successfully saved to {filepath}")


def load_weights(model, filepath):
    """
    Loads parameter arrays from disk and injects them directly
    into the model's memory addresses.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weight file '{filepath}' not found.")

    state_dict = np.load(filepath)
    params = model.get_parameters()

    if len(state_dict.files) != len(params):
        raise ValueError(
            f"Architecture mismatch! Disk contains {len(state_dict.files)} tensors, "
            f"but model requires {len(params)} tensors."
        )

    # Inject loaded data back into the original memory references
    for i, p in enumerate(params):
        p[:] = state_dict[f"param_{i}"]

    print(f"Model weights successfully loaded from {filepath}")