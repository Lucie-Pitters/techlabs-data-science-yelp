import os
import json
from collections import defaultdict

# Define the input file path (relative to your script location)
input_file = "data/data/yelp_academic_dataset_review.json"

# Dictionary to hold reviews by year and month
reviews_by_month = defaultdict(list)

# Read the input file and organize reviews by year and month
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        review = json.loads(line.strip())  # Parse JSON line
        review_date = review.get("date", "")
        if review_date:  # Ensure date is present
            year_month = review_date[:7]  # Extract "YYYY-MM" from date
            reviews_by_month[year_month].append(review)

# Create the output folder if it doesn't exist
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# Write each month's reviews to a separate file in the data folder
for year_month, reviews in reviews_by_month.items():
    output_file = os.path.join(output_folder, f"reviews_{year_month}.json")
    with open(output_file, "w", encoding="utf-8") as outfile:
        for review in reviews:
            json.dump(review, outfile)
            outfile.write("\n")  # Write each review on a new line

print(f"Splitting complete. Files saved in '{output_folder}' folder.")
