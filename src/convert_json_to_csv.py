import json
import pandas as pd
import os

# List of JSON files to process
json_files = [
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_checkin.json",
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_tip.json",
    "yelp_academic_dataset_user.json"
]

# Loop through each JSON file and convert it to CSV
for json_file in json_files:
    if os.path.exists(json_file):
        print(f"Processing {json_file}...")

        try:
            # Open the JSON file and read it line by line
            data = []
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))  # Convert each line to JSON object

            # Convert to DataFrame
            if data:
                df = pd.json_normalize(data)  # Normalize nested JSON structures

                # Define output CSV file name
                csv_file = json_file.replace(".json", ".csv")

                # Save to CSV file
                df.to_csv(csv_file, index=False, encoding='utf-8')

                print(f"Successfully saved {csv_file}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

print("All files have been converted successfully!")
