import os
import sys

def update_kaggle_dataset(dataset_folder, update_message):
    """
    Updates an existing Kaggle dataset using the Kaggle API.

    Parameters:
    - dataset_folder (str): Path to the folder containing dataset files and metadata.
    - update_message (str): Description of the update (e.g., "Added new JSON files").
    """
    if not os.path.exists(dataset_folder):
        print(f"Error: The folder '{dataset_folder}' does not exist.")
        sys.exit(1)

    # Check for dataset-metadata.json file
    metadata_path = os.path.join(dataset_folder, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        print("Error: dataset-metadata.json file is missing in the folder.")
        sys.exit(1)

    # Run Kaggle API command to update the dataset
    print("Updating Kaggle dataset...")
    command = f'kaggle datasets version -p "{dataset_folder}" -m "{update_message}"'
    exit_code = os.system(command)

    if exit_code == 0:
        print("Dataset successfully updated on Kaggle!")
    else:
        print("Error: Dataset update failed. Check the Kaggle API setup or dataset ID.")


# Edit the following variables as needed
dataset_folder = "data\Data"  # Replace with your dataset folder path
update_message = "Test update data set"  # Replace with your update message

# Call the update function
update_kaggle_dataset(dataset_folder, update_message)
