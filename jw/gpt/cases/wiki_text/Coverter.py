import json
import re
from pathlib import Path


# Specify the input JSON file and the output text file
# input_json_file = 'input/article_1.json'
# output_text_file = 'input/article_1_conv.txt'

def filter_text(text):
    trimmed = re.match(r'^[^=]*', text)
    return trimmed.group()


def text_eligible(text):
    return not re.search(r'[^a-zA-Z0-9\s.,!?\'\";:()\[\]{}<>@#$%^&*+=-]', text)


files_paths = []
folder = Path('input')
for file_path in folder.rglob('*.json'):
    files_paths.append(file_path)

with open('input/input_conv_4.txt', 'w', encoding='utf-8') as text_file:
    for file_path in files_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)  # Load JSON data as a Python object (list of dictionaries)

            for item in data:

                if 'title' in item and 'text' in item:
                    title = filter_text(item['title'])
                    text = filter_text(item['text'])
                    if text_eligible(title) and text_eligible(text):
                        text_file.write(f"{title}\n{text}\n\n")

        except Exception as e:
            print(f"An error occurred: {e}")
