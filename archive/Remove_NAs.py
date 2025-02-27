import json
import pandas as pd
import os

#File path
reviews_file = "M:\\OneDrive\\Dokumente\\GitHub\\techlabs-data-science-yelp\\data\\reviews_2021-01.json"
nas_file = "M:\\OneDrive\\Dokumente\\GitHub\\techlabs-data-science-yelp\\src\\NAs.json"

# Existins check for file
if not os.path.exists(reviews_file):
    print(f"Fehler '{reviews_file}' nicht gefunden")
    exit()

# JSON-file read in
with open(reviews_file, "r", encoding="utf-8") as file:
    reviews = [json.loads(line) for line in file]

# DataFrame birth
df = pd.DataFrame(reviews)

# finding NAs
nas = df[df.isnull().any(axis=1)]

# Save NAs
nas.to_json(nas_file, orient="records", lines=True, force_ascii=False)

# Delete NAs
df = df.dropna()

#Save Cleaned Data
df.to_json(reviews_file, orient="records", lines=True, force_ascii=False)

print("Missing values ​​removed and saved")
