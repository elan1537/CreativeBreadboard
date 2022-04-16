import os
import json

PATH_TO_FILES = "./images/Circuits/220414"
LABEL_TO_DELETE = ["vector", "structure"]

if __name__ == "__main__":
    files = [f for f in os.listdir(PATH_TO_FILES) if '.json' in f]

    for file in files:
        with open(f"{PATH_TO_FILES}/{file}", "r") as f:
            label_data = json.load(f)

            new_shape = [shape for shape in label_data['shapes'] \
                if "vector" not in shape['label'] and 'structure' not in shape['label']]

            label_data['shapes'] = new_shape

            with open(f"{PATH_TO_FILES}/modify_json/{file}", "w") as t:
                json.dump(label_data, t)