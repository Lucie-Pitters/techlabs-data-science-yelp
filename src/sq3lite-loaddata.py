import ijson
import sqlite3

with open("data/Data/yelp_academic_dataset_review.json", "r", encoding="utf-8") as infile:
    with open("data/Data/cleaned_yelp_academic_dataset_review.json", "w", encoding="utf-8") as outfile:
        for line in infile:
            # Strip any unnecessary spaces and check if it's valid JSON
            cleaned_line = line.strip()
            if cleaned_line:  # Only write non-empty lines
                outfile.write(cleaned_line + "\n")

# Connect to SQLite database
conn = sqlite3.connect('reviews.db')
cursor = conn.cursor()

# Open the JSON file using ijson
json_file = r"data/Data/cleaned_yelp_academic_dataset_review.json"

# Open the file and create a generator using ijson
with open(json_file, 'r', encoding="utf-8") as f:
    objects = ijson.items(f, '')  # Empty path means we start from the root
    
    # Iterate over the items (each "item" is a review record)
    for obj in objects:
        try:
            print("parse one object")
            # Extract fields (modify based on the actual structure of your JSON)
            review_id = obj.get('review_id', '')
            user_id = obj.get('user_id', 0)
            business_id = obj.get('business_id', '')
            stars = obj.get('stars', '')
            useful = obj.get('useful', '')
            funny = obj.get('funny', '')
            cool = obj.get('cool', '')
            text = obj.get('text', '')
            date = obj.get('date', '')
            # Insert data into SQLite
            cursor.execute('''
                INSERT INTO reviews (review_text, rating, date)
                VALUES (?, ?, ?)
            ''', (review_id, user_id, business_id, stars, useful, funny, cool, text, date))
        except Exception as e:
            print(f"Error parsing object: {e}")
            continue  # Skip malformed object

# Commit the changes and close the connection
conn.commit()
cursor.close()
conn.close()

print("Data successfully inserted into SQLite.")
