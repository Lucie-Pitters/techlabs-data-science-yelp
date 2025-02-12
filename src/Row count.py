import os
import pandas as pd

# Define the folder where the dataset files are located
folder_path = "D:/TechLab Project/Yelp Dataset"  # Update this to the correct path

# List of dataset file names
file_names = [
    "yelp_academic_dataset_business.csv",
    "yelp_academic_dataset_checkin.csv",
    "yelp_academic_dataset_review.csv",
    "yelp_academic_dataset_tip.csv",
    "yelp_academic_dataset_user.csv"
]

# Loop through each file and count the number of rows
for file in file_names:
    file_path = os.path.join(folder_path, file)  # Create full file path
    
    if os.path.exists(file_path):  # Check if the file exists
        try:
            # Load the dataset
            df = pd.read_csv(file_path, low_memory=False)
            
            # Count the number of rows
            row_count = len(df)
            
            # Print the result
            print(f"Number of rows in {file}: {row_count}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    else:
        print(f"File not found: {file_path}")

