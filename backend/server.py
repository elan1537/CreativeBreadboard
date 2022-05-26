import os, json, cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_cors import CORS
from findColor import test
from findResistor import toPerspectiveImage, checkResistor
import requests
import base64
from shutil import copy
from diagram import drawDiagram

circuit_component_data = [
    [
        { "name": "R10", "value": 50 },
    ],
    [       
        { "name": "R20", "value": 50 },
        { "name": "R21", "value": 50 }
    ],
    [
        { "name": "R30", "value": 100 },
        { "name": "R31", "value": 100 },
        { "name": "R32", "value": 3000 }
    ]
]

SAVE_PATH = "./static/uploads"
PROJECT_PATH = "/Users/se_park/Library/Mobile Documents/com~apple~CloudDocs/2022 Soongsil/1. CS/CreativeBreadboard/images/Circuits"

# jpg_original = base64.b64decode(img_data)
# img_arr = np.frombuffer(jpg_original, np.uint8)
# target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

# cv2.imwrite("result_from_post.jpg", target_image)

app = Flask(__name__, static_folder="./static", template_folder="./templates")
# app.config.from_object(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

FILE_IMAGE = None

@app.route("/")
def main():
    global FILE_IMAGE
    if FILE_IMAGE:
        print(FILE_IMAGE)
        return render_template("image.html", image_path = FILE_IMAGE)
    else:
        return render_template("image.html")

@app.route("/resistor", methods=['GET', 'POST'])
def resistor():
    if request.method == 'GET':
        return jsonify({
            "state": "success",
            "data": circuit_component_data
        })
    if request.method == 'POST':
        resistor_value = request.get_json()

        for r in resistor_value:
            for row in circuit_component_data:
                for col in row:
                    if r['name'] == col['name']:
                        col['value'] = int(r['value'])

        return jsonify({
            "state": "success"
        })

@app.route("/draw", methods=["GET"])
def draw():
    if request.method == 'GET':
        image_bytes = drawDiagram(5, circuit_component_data)
        return jsonify({
            "state": "success",
            "circuit": base64.b64encode(image_bytes).decode()
        })

@app.route("/image", methods=['POST'])
def image():
    if request.method == 'POST':
        global FILE_IMAGE
        img_file = request.files['image']
        data = json.load(request.files['data'])

        img_file_bytes = img_file.stream.read()
        img_arr = np.frombuffer(img_file_bytes, np.uint8)
        target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        points = data["points"]
        scale = float(data["scale"])

        pts = []
        for point in points:
            pts.append([int(point[0] / scale), int(point[1] / scale)])

        base_point, res = toPerspectiveImage(target_image, np.array(pts), 100)

        name = data["img_name"].replace(".jpeg", "").replace(".jpg", "").replace(".JPG" ,"")

        # 딥러닝 데이터셋 추가 시작
        filepath = findfile(f"{name}.json", PROJECT_PATH)
        json.dump(pts, open(f"./static/uploads/check_points/{name}.json", "w"))
        copy(filepath, "/Users/se_park/Library/Mobile Documents/com~apple~CloudDocs/2022 Soongsil/1. CS/CreativeBreadboard/backend/static/uploads/annotation")
        cv2.imwrite(f"./static/uploads/origin_img/{data['img_name']}", target_image)
        # 딥러닝 데이터셋 추가 끝

        _, buffer = cv2.imencode('.jpg', res)
        jpg_as_text = base64.b64encode(buffer).decode()
        res = requests.post("http://localhost:3000/getResistor", json=json.dumps({'pts': base_point.tolist(), 'img_res': jpg_as_text}))
        
        img_data = res.json()

        return jsonify(img_data)

@app.route("/points", methods=['POST'])
def points():
    data = json.load(request.files['data'])
    points = data["points"]
    img_name = data["img_name"]
    print(img_name, points)

    return jsonify({"message": "success"})

@app.route("/getResistor", methods=['POST'])
def getResistor():
    data = json.loads(request.get_json())
    pts = data['pts']
    img_res = data['img_res']
    jpg_original = base64.b64decode(img_res)
    img_arr = np.frombuffer(jpg_original, np.uint8)
    target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    get_resistor_picking_image = checkResistor(target_image, pts)

    _, buffer = cv2.imencode('.jpg', get_resistor_picking_image)
    jpg_as_text = base64.b64encode(buffer).decode()

    return jsonify({
        "result_image": jpg_as_text
    })

@app.route('/result', methods=["GET"])
def result():
    test()
    return send_file('./images/precess_image.jpg', mimetype='image/jpg')

def check():
    files = os.listdir('./static/uploads')
    print(files)

    if files:
        print("files ok")
        if '.jpg' in files[-1] or '.jpeg' in files[-1] or '.png' in files[-1]:
            global FILE_IMAGE
            FILE_IMAGE = files[-1]
            print("asdf", FILE_IMAGE)

def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)

if __name__ == "__main__":
    check()
    app.run(debug=True, host='0.0.0.0', port=3000)