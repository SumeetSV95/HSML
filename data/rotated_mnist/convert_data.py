import torch
import numpy as np
import pickle
import os

def convert_pt_to_full_pkl(pt_file_path, output_dir):
    """
    Loads a PyTorch .pt file, converts all tensors to NumPy arrays,
    and saves the full dataset in a TensorFlow-compatible .pkl format.
    This version does NOT create support/query splits; it saves all data.
    """
    if not os.path.exists(pt_file_path):
        print(f"Error: PyTorch data file not found at {pt_file_path}")
        return

    print(f"Loading PyTorch data from {pt_file_path}...")
    train_dataset_pt, test_dataset_pt = torch.load(pt_file_path)
    print("PyTorch data loaded successfully.")

    def process_full_dataset(dataset_pt):
        """Processes a list of tasks, converting tensors to numpy arrays."""
        processed_data_list = []
        for i, task_data in enumerate(dataset_pt):
            angle, all_x_pt, all_y_pt = task_data
            
            # Convert the full tensors to numpy arrays
            all_x_np = all_x_pt.numpy()
            all_y_np = all_y_pt.numpy()

            # Store the full data for the task
            processed_data_list.append((all_x_np, all_y_np))
        return processed_data_list

    # Process both train and test datasets fully
    print("Processing training data...")
    train_data_full = process_full_dataset(train_dataset_pt)
    print("Processing testing data...")
    test_data_full = process_full_dataset(test_dataset_pt)

    # The final structure will be a tuple: (train_list, test_list)
    # This matches the structure of the original HSML data file we discussed.
    final_data_structure = (train_data_full, test_data_full, test_data_full) # Using test for validation as well

    # Save to a single .pkl file
    output_path = os.path.join(output_dir, 'mnist_all_rotation_normalized_train_valid.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(final_data_structure, f)
    print(f"Successfully saved full converted data to {output_path}")


if __name__ == "__main__":
    # File paths
    base_dir = os.path.dirname(__file__)
    pt_path = os.path.join(base_dir, 'data', 'mnist_rotations.pt')
    output_dir = base_dir # Save .pkl file in the /data/rotated_mnist/ directory

    # Run conversion
    convert_pt_to_full_pkl(pt_path, output_dir)