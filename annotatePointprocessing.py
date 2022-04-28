import os
import json

PATH_TO_FILES = "./images/Circuits/220421"

if __name__ == "__main__":
    files = [f for f in os.listdir(PATH_TO_FILES) if '.json' in f]

    # "modify_json" 폴더 없으면 생성
    if not os.path.isdir(f"./{PATH_TO_FILES}/modify_json"):
        os.mkdir(f"{PATH_TO_FILES}/modify_json")   

    # 이미 "modify_json" 폴더가 존재하면 안에 있던 기존 파일들 제거
    else: 
        for file in os.listdir(f"./{PATH_TO_FILES}/modify_json"):
            os.remove(f"./{PATH_TO_FILES}/modify_json/{file}")

    for file in files:
        with open(f"{PATH_TO_FILES}/{file}", "r") as f:
            label_data = json.load(f)

            # *vector, *structure 라벨링만 제외한 것들을 new_shape로 담아
            new_shape = [shape for shape in label_data['shapes'] \
                if "vector" not in shape['label'] and "structure" not in shape["label"]]

            label_data['shapes'] = new_shape

            # 새로운 파일로 생성함
            with open(f"{PATH_TO_FILES}/modify_json/{file}", "w") as t:
                json.dump(label_data, t)