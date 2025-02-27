import json
import pandas as pd
import os

# Step 1: Find the "base directory" (go 2 levels up from /src1/)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Step 2: Define input and output directories using relative paths
input_dir = os.path.join(base_dir, "data", "raw")
output_dir = os.path.join(base_dir, "data", "intermediate")

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

def convert_json_to_csv(json_file):
    json_file_path = os.path.join(input_dir, json_file)
    if os.path.exists(json_file_path):
        print(f"Processing {json_file_path}...")

        try:
            data = []
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

            if data:
                df = pd.json_normalize(data)

                # Define output file path
                csv_file = os.path.join(output_dir, json_file.replace(".json", ".csv"))

                # Save DataFrame to CSV
                df.to_csv(csv_file, index=False, encoding='utf-8')

                print(f"Successfully saved {csv_file}")

        except Exception as e:
            print(f"Error processing {json_file_path}: {e}")

def convert_multiple_json_to_csv(json_files):
    for json_file in json_files:
        convert_json_to_csv(json_file)
    print("All files have been converted successfully!")


