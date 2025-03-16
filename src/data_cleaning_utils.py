import json
import re
import pandas as pd

def load_json_lines(filepath):
    """Load JSON file with one JSON object per line."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def save_json_lines(data, filepath):
    """Save data as JSONL (line-delimited JSON)."""
    with open(filepath, 'w', encoding='utf-8') as file:
        for record in data:
            json.dump(record, file, ensure_ascii=False)
            file.write('\n')

def clean_text(text):
    """Text cleaning: lowercase & remove special characters."""
    if pd.isnull(text):
        return text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

