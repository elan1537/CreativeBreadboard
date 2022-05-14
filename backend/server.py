import os, json, cv2
import numpy as np
from flask import Flask, redirect, render_template, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from findColor import toPerspectiveImage, test

app = Flask(__name__)
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
        f = request.files['file']
        f.save('./static/uploads/' + secure_filename(f.filename))

        FILE_IMAGE = f.filename
        return redirect("/")

@app.route('/result', methods=["GET"])
def result():
    test()
    return send_file('./images/precess_image.jpg', mimetype='image/jpg')

@app.route("/points", methods=["POST"])
def points():
    if request.method == "POST":
        points = json.loads(request.data)['points']
        scale = float(json.loads(request.data)['scale'])

        pts = []
        for point in points:
            pts.append([int(point[0] / scale), int(point[1] / scale)])

        target_image = cv2.imread(f"./static/uploads/{FILE_IMAGE}")
        _, res = toPerspectiveImage(target_image, np.array(pts))

        cv2.imwrite("./images/res.jpg", res)

        return "Success"
    else:
        return "Bad access"

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