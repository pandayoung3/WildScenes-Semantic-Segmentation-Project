import os
import random


def split_dataset(dataset_folder, train_ratio=0.8):
    # Get list of all files in the dataset folder
    all_files = os.listdir(dataset_folder)

    # Shuffle the files
    random.shuffle(all_files)

    # Calculate the number of training samples
    train_size = int(len(all_files) * train_ratio)

    # Split the files into training and validation sets
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]

    # Write the training files to train.txt
    with open(os.path.join(dataset_folder, 'train.txt'), 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")

    # Write the validation files to val.txt
    with open(os.path.join(dataset_folder, 'val.txt'), 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")

    print(f"Dataset split completed: {len(train_files)} training samples, {len(val_files)} validation samples.")


# Example usage
dataset_folder = 'OneDrive_1_2024-7-25/K-03/image'  # Change this to your dataset folder path
split_dataset(dataset_folder)
