import json
import re

# Function to remove special characters
def remove_special_chars(text):
    # Use regex to remove any non-alphanumeric characters (except spaces)
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Load the JSON data
with open('D:/TechLab Project/techlabs-data-science-yelp/data/reviews_2021-01.json', 'r') as file:
    data = [json.loads(line) for line in file]

# Process each review
for review in data:
    if 'text' in review:
        # Convert to lowercase
        review['text'] = review['text'].lower()
        # Remove special characters
        review['text'] = remove_special_chars(review['text'])

# Save the modified data to a new JSON file
with open('D:/TechLab Project/techlabs-data-science-yelp/data/reviews_2021-01_textcleaning.json', 'w') as file:
    for review in data:
        json.dump(review, file)
        file.write('\n')

print("Processing complete. Data saved to 'reviews_2021-01_processed.json'.")