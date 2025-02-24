import json
import csv

# Function to read reviews from a JSON file that may contain multiple JSON objects
def lese_bewertungen(file_name):
    reviews = []
    with open(file_name, 'r', encoding='utf-8') as file:
        try:
            # Try to load the JSON data, which may contain multiple objects
            data = file.read().strip()
            # If the file contains multiple JSON objects, split by the newline
            if data:
                for line in data.splitlines():
                    try:
                        review = json.loads(line)  # Parse each line as a separate JSON object
                        reviews.append(review['text'])  # Assuming each review has a 'text' field
                    except json.JSONDecodeError:
                        continue  # Skip any line that doesn't decode correctly
        except json.JSONDecodeError as e:
            print(f"Error loading JSON data: {e}")
    return reviews

# Function to categorize reviews based on keywords
def kategorisiere_bewertungen(bewertungen):
    customer_service_reviews = []
    service_reviews = []

    customer_service_keywords = ['friendly', 'helpful', 'service', 'support', 'customer service']
    service_keywords = ['product', 'quality', 'price', 'good', 'offer']

    for bewertung in bewertungen:
        review_lower = bewertung.lower()
        if any(keyword in review_lower for keyword in customer_service_keywords):
            customer_service_reviews.append(bewertung.strip())
        elif any(keyword in review_lower for keyword in service_keywords):
            service_reviews.append(bewertung.strip())
        else:
            customer_service_reviews.append('')
            service_reviews.append('')
    
    return customer_service_reviews, service_reviews

# Function to save reviews into a CSV file
def speichere_csv(customer_service_reviews, service_reviews, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Customer Service', 'Service'])
        for service_review, product_review in zip(customer_service_reviews, service_reviews):
            writer.writerow([service_review, product_review])

# Main code execution
input_file = r'C:\Users\moham\OneDrive\Dokumente\GitHub\techlabs-data-science-yelp\data\raw\reviews_2021-01.json'  # Update the path
reviews = lese_bewertungen(input_file)

customer_service_reviews, service_reviews = kategorisiere_bewertungen(reviews)

output_file = 'reviews.csv'
speichere_csv(customer_service_reviews, service_reviews, output_file)

print("Reviews have been successfully categorized and saved into a CSV file.")
