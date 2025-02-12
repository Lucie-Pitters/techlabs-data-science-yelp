import pandas as pd
import re

def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s]+', '', text)

try:
    # Load the dataset from JSON
    json_file_path = "D:/TechLab Project/techlabs-data-science-yelp/data/reviews_2021-01.json"  # Replace with actual file path
    df_json = pd.read_json(json_file_path, dtype=str, lines=True)  # Load all columns as strings

    # Convert all text data to lowercase and remove special characters except the first column (assumed to be 'business_id')
    df_json.iloc[:, 1:] = df_json.iloc[:, 1:].applymap(lambda x: remove_special_characters(x.lower()) if isinstance(x, str) else x)

    # Save the modified dataset back to JSON
    modified_json_file_path = "D:/TechLab Project/techlabs-data-science-yelp/data/reviews_2021-01_textcleaning.json"
    df_json.to_json(modified_json_file_path, orient="records", lines=True)

    print(f"Modified file saved at: {modified_json_file_path}")

except FileNotFoundError:
    print(f"File not found: {json_file_path}")
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")