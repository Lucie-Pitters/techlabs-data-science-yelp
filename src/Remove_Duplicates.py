import json
import pandas as pd
import os

#File path
reviews_file = "M:\\OneDrive\\Dokumente\\GitHub\\techlabs-data-science-yelp\\data\\reviews_2021-01.json"
duplicates_file = "M:\\OneDrive\\Dokumente\\GitHub\\techlabs-data-science-yelp\\src\\Duplicates.json"

# Existins check for file
if not os.path.exists(reviews_file):
    print(f"Error '{reviews_file}' not found")
    exit()

# JSON-file read in
with open(reviews_file, "r", encoding="utf-8") as file:
    reviews = [json.loads(line) for line in file]

# DataFrame birth
df = pd.DataFrame(reviews)

# finding duplicates through review_id
duplicates = df[df.duplicated(subset=['review_id'], keep=False)]

# Save duplicates
duplicates.to_json(duplicates_file, orient="records", lines=True, force_ascii=False)

# Remove Duplicates from review file
df = df.drop_duplicates(subset=['review_id'], keep="first")

# Save cleaned file
df.to_json(reviews_file, orient="records", lines=True, force_ascii=False)

print("Duplicates removed and saved")
