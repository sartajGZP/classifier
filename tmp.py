import json
import csv
import os

with open('data/labels.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

with open('dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    
    # Header: Removed file_path
    writer.writerow(['tags', 'text'])
    
    for path, tags in data.items():
        # Joins multiple tags like ["corruption", "Vote Vapasi"] into "corruption, Vote Vapasi"
        tags_string = ", ".join(tags)
        
        file_text = ""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as text_file:
                # Flattens all multi-line text into a single line
                file_text = text_file.read().replace('\n', ' ').replace('\r', ' ').strip()
                
        # Write only the tags and the text
        writer.writerow([tags_string, file_text])

print("Dataset generated successfully without file paths!")

