import os, json, cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from findColor import test
from findResistor import toPerspectiveImage, checkResistor
import requests
import base64

SAVE_PATH = "./static/uploads"

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

        cv2.imwrite(f"./static/{data['img_name']}", res)

        _, buffer = cv2.imencode('.jpg', res)
        jpg_as_text = base64.b64encode(buffer).decode()
        res = requests.post("http://137.184.95.69:3000/getResistor", json=json.dumps({'pts': base_point.tolist(), 'img_res': jpg_as_text}))
        
        img_data = res.json()
        # jpg_original = base64.b64decode(img_data)
        # img_arr = np.frombuffer(jpg_original, np.uint8)
        # target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # cv2.imwrite("result_from_post.jpg", target_image)

        return jsonify(img_data)

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

if __name__ == "__main__":
    check()
    app.run(debug=True, host='0.0.0.0', port=3000)