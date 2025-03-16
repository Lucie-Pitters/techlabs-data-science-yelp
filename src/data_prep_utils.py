import os
import json
from collections import defaultdict

# Function to read JSON file line by line
def read_json_file(input_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            yield json.loads(line.strip())  # Yield each review as a dict

# Function to organize reviews by year/month or just year
def organize_reviews_by_time(input_file, time_format="month"):
    reviews_by_time = defaultdict(list)

    for review in read_json_file(input_file):
        review_date = review.get("date", "")
        if review_date:  # Ensure date is present
            if time_format == "month":
                time_key = review_date[:7]  # Extract "YYYY-MM"
            elif time_format == "year":
                time_key = review_date.split("-")[0]  # Extract "YYYY"
            reviews_by_time[time_key].append(review)

    return reviews_by_time

# Function to save reviews into separate files
def save_reviews_by_time(reviews_by_time, output_folder, time_format="month"):
    os.makedirs(output_folder, exist_ok=True)

    for time_key, reviews in reviews_by_time.items():
        output_file = os.path.join(output_folder, f"reviews_{time_key}.json")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for review in reviews:
                json.dump(review, outfile)
                outfile.write("\n")  # Write each review on a new line

    print(f"Splitting complete. Files saved in '{output_folder}' folder.")
