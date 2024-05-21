import os
import json


def extract_unique_classes(directory):
    unique_classes = set()  # A set to store unique classes
    for filename in os.listdir(directory):
        if filename.endswith('.json'):  # Make sure to process only JSON files
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                for item in data['arr_boxes']:
                    unique_classes.add(item.get('genre'))  # Add class to the set
    return unique_classes


if __name__ == "__main__":
    directory_path = '~/Downloads/data'
    classes = extract_unique_classes(directory_path)
    print(classes)
