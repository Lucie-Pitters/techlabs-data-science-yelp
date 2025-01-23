import os
import json
from collections import defaultdict



def split_json_file(input_file, output_folder, output_file1, output_file2):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the file line by line
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Split the lines into two parts
    mid_index = len(lines) // 2
    part1_lines = lines[:mid_index]
    part2_lines = lines[mid_index:]

    # Save the two parts to separate files
    output_file1 = os.path.join(output_folder, output_file1)
    with open(output_file1, 'w', encoding='utf-8') as file1:
        file1.writelines(part1_lines)   
    output_file2 = os.path.join(output_folder, output_file2)
    with open(output_file2, 'w', encoding='utf-8') as file2:
        file2.writelines(part2_lines)
        
    print(f"JSON file split into '{output_file1}' and '{output_file2}'")


input_file = "data/yelp_academic_dataset_user.json"
output_folder = "data"
output_file1 = "yelp_user1.json"
output_file2 = "yelp_user2.json"
split_json_file(input_file, output_folder, output_file1, output_file2)
