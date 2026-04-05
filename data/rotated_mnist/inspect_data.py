import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def save_sample(x, y, t, output_dir, prefix):
    """Saves a single image sample to disk."""
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.item()

    img = x.reshape(28, 28)
    label = y
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Label: {label}, Task Angle: {t:.1f}°")
    angle_str = str(t).replace('.', '_')
    plt.savefig(os.path.join(output_dir, f"{prefix}_task_angle_{angle_str}_label_{label}.png"))
    plt.close()

def inspect_pytorch_data(data_file_path, output_dir):
    """Loads a .pt file, prints its structure, and finds all unique task angles."""
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    torch.manual_seed(0)
    try:
        train_dataset, test_dataset = torch.load(data_file_path)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return

    print("--- Data Structure Analysis ---")
    
    num_train_tasks = len(train_dataset)
    print(f"Number of training tasks: {num_train_tasks}")
    
    if num_train_tasks > 0:
        # --- Correctly unpack the data based on the deduced structure ---
        task_angle, all_x, all_y = train_dataset[0]
        
        # --- Manually split into support and query sets for analysis ---
        # Let's assume a 10-shot scenario for inspection
        k_shot = 10
        support_x, query_x = all_x[:k_shot], all_x[k_shot:]
        support_y, query_y = all_y[:k_shot], all_y[k_shot:]

        print("\nStructure of a single training task:")
        print(f"  - Task Identifier (Angle): {task_angle}")
        print(f"  - Total samples in task: {len(all_x)}")
        print(f"  - Inferred Support Set shapes (x, y): ({support_x.shape}, {support_y.shape})")
        print(f"  - Inferred Query Set shapes (x, y): ({query_x.shape}, {query_y.shape})")

        # --- Find all unique angles in the dataset ---
        all_angles = sorted(list(set([task[0] for task in train_dataset])))
        print(f"\nFound {len(all_angles)} unique rotation angles in the training set:")
        print([round(a, 2) for a in all_angles]) # Print rounded angles for readability

        # --- Save sample images from the inferred support and query sets ---
        print("\n--- Saving Sample Images ---")
        n_samples_to_save = 3
        for i in range(n_samples_to_save):
            save_sample(support_x[i], support_y[i], task_angle, output_dir, prefix="train_support")
            save_sample(query_x[i], query_y[i], task_angle, output_dir, prefix="train_query")
        print(f"Saved {n_samples_to_save*2} sample images to '{output_dir}'")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'mnist_rotations.pt')
    output_path = os.path.join(os.path.dirname(__file__), 'sample_images')
    inspect_pytorch_data(data_path, output_path)