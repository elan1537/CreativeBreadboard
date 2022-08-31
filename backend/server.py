import datetime
import os, json, cv2
import requests
import base64
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from findComponents import *
from shutil import copy
from data_processing.diagram import drawDiagram
from data_processing.calcVoltageAndCurrent import calcCurrentAndVoltage
from component_predict import get_component
from tqdm import tqdm


pd.set_option("mode.chained_assignment", None)

circuit_component_data = []
body_pinmap = None
vol_pinmap = None
pinmap = None
pinmap_shape = None
start_point = None

find_pincoords_resi_model = None
find_pincoords_line_model = None

V = 5
# PADDING = 0

MODEL_RESISTORAREA_PATH = "../model/resistor-area.model.pt"
MODEL_RESISTORBODY_PATH = "../model/resistor.body.pt"
MODEL_LINEAREA_PATH = "../model/line-area.model.pt"
MODEL_LINEENDAREA_PATH = "../model/line-endpoint.model.pt"

app = Flask(__name__, static_folder="./static")
app.secret_key = "f#@&v08@#&*fnvn"
app.permanent_session_lifetime = datetime.timedelta(hours=4)

CORS(app, resources={r"/*": {"origins": "*"}})

FILE_IMAGE = None
models = {}


@app.route("/")
def main():
    """
    서버 동작 체크 여부
    """
    return "Hello, I'm on"


@app.route("/pinmap", methods=["GET"])
def pinmap():
    """
    브레드보드에서의 Pin위치를 전달하면 입력된 이미지 기준으로 해당하는 Pin의 x, y 픽셀 좌표를 반환한다.
    Pinmap은 backend/static/data/pinmap.json의 파일을 기준으로 한다.
    Pinmap은 사용자가 선택한 4개의 꼭짓점으로 변환된 이미지에 mapping한다. 이미지따라 핀의 픽셀 위치가 다 다르기 때문이다.
    ex) A01 => 130, 245
    ex) V225 => 110, 3000
    """
    global body_pinmap, vol_pinmap
    search_map = None
    row = None
    col = None
    search_pin = request.args.get("pin")

    assert search_pin != None
    assert search_pin[0] in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "V"]

    if search_pin[0] == "V":
        assert int(search_pin[2:]) >= 1 and int(search_pin[2:]) <= 25
        row, col = int(search_pin[2:]) - 1, search_pin[:2]
    else:
        assert int(search_pin[1:]) >= 1 and int(search_pin[1:]) <= 30
        row, col = int(search_pin[1:]) - 1, search_pin[0]

    search_map = pd.concat(
        [vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1
    )

    print(search_map)
    x, y = search_map.xs(row)[col]["x"], search_map.xs(row)[col]["y"]

    return jsonify({"coord": [x, y]})


@app.route("/resistor", methods=["GET", "POST"])
def resistor():
    """
    To-do: 만약 저항값이 잘못 검측되었을 때 필요로 함
    분석이 완료된 회로 계층의 저항값을 반환받는다.
    Ex) 회로가 1 - 1 - 2(병렬 저항)로 구성됨
    response = {
        "state": "success",
        "data": [[{'name': 'R0', 'value': 100}], [{'name': 'R1', 'value': 100}], [{'name': 'R2', 'value': 100}, {'name': 'R3', 'value': 100}]]
    }
    GET: 서버 메모리에 올라간 회로 계층 데이터 반환
    POST: 전달된 데이터를 바탕으로 저항값 데이터 초기화
    """
    global circuit_component_data

    if request.method == "GET":
        return jsonify({"state": "success", "data": circuit_component_data})

    if request.method == "POST":
        resistor_value = request.get_json()

        for r in resistor_value:
            for row in circuit_component_data:
                for col in row:
                    if r["name"] == col["name"]:
                        col["value"] = int(r["value"])

        return jsonify({"state": "success"})


@app.route("/draw", methods=["GET"])
def draw():
    """
    회로 다이어그램 그림을 반환함
    """
    if request.method == "GET":
        print(circuit_component_data)
        image_bytes = drawDiagram(V, circuit_component_data)
        return jsonify(
            {"state": "success", "circuit": base64.b64encode(image_bytes).decode()}
        )


@app.route("/image", methods=["POST"])
def image():
    """
    /upload에서 업로드한 이미지 데이터와 전압 데이터를 전달받는다.
    이후 /detect를 호출하여 전기소자 예측 모델과 전기소자 위치 검출 모델로 데이터를 반환받는다.

    """
    if request.method == "POST":
        global FILE_IMAGE, V
        PADDING = 200
        target_image = None
        points = None
        scale = None

        if request.files:
            data = json.loads(request.form["points"])
            circuitImage = request.files["circuitImage"].read()
            target_image = cv2.imdecode(
                np.frombuffer(circuitImage, np.uint8), cv2.IMREAD_COLOR
            )
            points = data["points"]
            scale = float(data["scale"])
            V = int(data["voltage"])

        # 테스트 데이터를 위한 분기
        else:
            target_image = cv2.imread("../IMG_5633.JPG", cv2.IMREAD_COLOR)
            points = [[93, 29], [99, 871], [648, 865], [648, 27]]
            scale = 0.25
            V = 15

        # 전달받은 4개의 포인트는 스케일이 적용되어 있다.
        # 웹에서 포인트를 선택하는 영역은 화면의 크기에 따라 해당하는 점 위치가 다르기 때문에
        # 실제 이미지 크기에 맞게 스케일링을 한다.
        # 현재 scale은 0.25로 고정되어있다.

        pts = []
        for point in points:
            pts.append([int(point[0] / scale), int(point[1] / scale)])

        base_point, target_image = toPerspectiveImage(
            target_image, np.array(pts), PADDING
        )
        cv2.imwrite("./target_image.jpg", target_image)

        _, buffer = cv2.imencode(".jpg", target_image)
        transformedImg_base64 = base64.b64encode(buffer).decode()

        # 해당 부분에서 검출 메소드를 호출한다.
        # res = requests.post(
        #     "http://localhost:3000/detect",
        #     json=json.dumps(
        #         {"pts": base_point.tolist(), "img_res": jpg_as_text, "scale": scale}
        #     ),
        # )
        component = detect(
            pts=base_point,
            target_image=target_image,
            scale=scale,
        )

        # components = res.json()

        result_data = {
            "transformedImg": transformedImg_base64,
            "basePoint": base_point.tolist(),
            "voltage": V,
            "scale": 0.25,
            "components": component["components"],
        }

        return jsonify(result_data)


@app.route("/calc", methods=["get"])
def calc():
    """
    분석된 회로에서의 이론적인 노드 전압과 출력 전류 그리고 합성저항값을 반환받는다.
    """
    global circuit_component_data
    R_TH, I, NODE_VOL = calcCurrentAndVoltage(V, circuit_component_data)

    return jsonify(
        {
            "circuit_analysis": {
                "r_th": str(R_TH),
                "node_current": str(I),
                "node_voltage": str(NODE_VOL),
            }
        }
    )


@app.route("/network", methods=["GET", "POST"])
def network():
    """
    네트워크의 계층을 찾지않고 검출된 순서대로 정렬된 각 전기소자들을 회로가 결선된 순서에 맞게 레이어를 구성한다.
    """
    if request.method == "POST":
        global circuit_component_data
        components = request.get_json()

        lines = pd.DataFrame(components["Line"])
        resistors = pd.DataFrame(components["Resistor"])

        components = pd.concat([lines, resistors], axis=1).transpose()

        print(components)

        """
            일단 지금은 사용자가 모든 핀을 정상적으로 오류를 캐치했을 때를 가정하고 네트워크를 구성하고 있음
            추가적으로 네트워크를 찾다가 잘못된 부분을 alert 하는거 구현해야함.
        """
        """
            OMG... D7과 H7이 같은 노드라고 나옴.. 이거 분리..~
        """

        circuit = findNetwork(components)

        # 아래의 코드는 위에서 계층을 찾으면 그걸 토대로 저항소자만 빼오는 코드
        # 위 데이터의 결과는 [[{'name': 'R0', 'value': 100}], [{'name': 'R1', 'value': 100}], [{'name': 'R2', 'value': 100}, {'name': 'R3', 'value': 100}]]

        print(circuit)

        table = {}

        for i in range(len(circuit)):
            row = circuit.iloc[i]

            if "R" not in row["name"]:
                continue

            d = {"name": row["name"], "value": row["value"]}

            if table.get(row.layer) is None:
                table[int(row.layer)] = [d]
            else:
                table[int(row.layer)].append(d)

        table = [value for _, value in table.items()]

        circuit_component_data = table

        return jsonify({"network": table})


# @app.route("/detect", methods=["POST"])
def detect(pts: np.ndarray, target_image: np.ndarray, scale: float):
    """
    전달받은 이미지 데이터에서 전기소자를 예측하고 그것의 위치를 찾는다.
    """
    # 이미지 프로세싱
    global circuit_component_data, vol_pinmap, body_pinmap, start_point, find_pincoords_resi_model, find_pincoords_line_model

    init()  # 검출을 위해 데이터 초기화 및 모델 로딩(로딩이 되어있지 않다면)

    # data = json.loads(request.get_json())
    # pts = data["pts"]
    # img_res = data["img_res"]
    # scale = data["scale"]
    # jpg_original = base64.b64decode(img_res)
    # img_arr = np.frombuffer(jpg_original, np.uint8)
    # target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    canvas_image = target_image.copy()

    cv2.imwrite("./target_img.jpg", canvas_image)

    resistor_area_pd = detectArea(
        models["resistor.area"], "resistor-area", target_image, 0.5
    )
    resistor_body_pd = detectArea(
        models["resistor.body"], "resistor-body", target_image, 0.5
    )
    line_area_pd = detectArea(models["line.area"], "line-area", target_image, 0.5)
    line_endarea_pd = detectArea(
        models["line.endpoint"], "line-endpoint", target_image, 0.5
    )

    base_point = np.array(pts, np.float32)
    transform_mtrx = cv2.getPerspectiveTransform(start_point, base_point)

    initializePinmaps(
        body_pinmap, vol_pinmap, transform_mtrx
    )  # 왜곡된 이미지에 맞게 핀맵의 좌표도 똑같이 변환
    search_map = pd.concat(
        [vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1
    )

    print("resistor_area_pd:", len(resistor_area_pd))
    print("resistor_body_pd:", len(resistor_body_pd))
    print("line_area_pd:", len(line_area_pd))
    print("line_endarea_pd:", len(line_endarea_pd))

    components = {"Resistor": [], "Unknown": []}

    detected_resistor_components = set_resistor_component(
        resistor_area_pd,
        search_map,
        base_point,
        target_image,
        canvas_image,
        body_pinmap,
        vol_pinmap,
        models["resistor.pin"],
    )
    components["Resistor"] = detected_resistor_components

    cv2.imwrite("canvas_image.jpg", canvas_image)

    # 전선영역이 검출되었다면
    if len(line_area_pd) > 0:
        table = line_contains_table(line_area_pd, line_endarea_pd)

        detected_line_components = set_line_component(
            line_area_pd,
            line_endarea_pd,
            table,
            search_map,
            base_point,
            target_image,
            canvas_image,
            body_pinmap,
            vol_pinmap,
            models["line.pin"],
        )
        components["Line"] = detected_line_components["Line"]
        components["Unknown"] = detected_line_components["Unknown"]

        for lineAreaIdx in table.values():
            r = int(random.random() * 255)
            g = int(random.random() * 255)
            b = int(random.random() * 255)

            linearea = line_area_pd.iloc[lineAreaIdx]

            lineareaMinPoint = round(linearea.xmin) - 30, round(linearea.ymin) - 30
            lineareaMaxPoint = round(linearea.xmax) + 30, round(linearea.ymax) + 30

            cv2.putText(
                canvas_image,
                f"linearea#{lineAreaIdx}",
                (lineareaMinPoint[0], lineareaMinPoint[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 255),
                2,
            )
            cv2.rectangle(
                canvas_image, lineareaMinPoint, lineareaMaxPoint, (b, g, r), 10
            )
    else:
        components["Line"] = {}

    """
    # 잘못 짜여짐
    # 전선 영역은 검출되지 않고 전선 꼭지만 검출되거나 영역 관계를 확인하지 못하는 컴포넌트들을
    # 따로 담아두고 웹에서 보정하게끔 해야함
    if len(line_endarea_pd) > 0:
        for lineEndAreaIdx in range(len(line_endarea_pd)):
            r = int(random.random() * 255)
            g = int(random.random() * 255)
            b = int(random.random() * 255)

            linearea = line_endarea_pd.iloc[lineEndAreaIdx]

            lineareaMinPoint = [round(linearea.xmin) - 30, round(linearea.ymin) - 30]
            lineareaMaxPoint = [round(linearea.xmax) + 30, round(linearea.ymax) + 30]

            if lineareaMinPoint[0] < 0:
                lineareaMinPoint[0] = 0
            if lineareaMinPoint[1] < 0:
                lineareaMinPoint[1] = 0

            expand_to = 350

            area_start, area_end, area = area_padding(target_image, lineareaMinPoint, lineareaMaxPoint, base_point[0], base_point[2], expand_to, True)
            table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)
            normalized = imgNormalizing(area, scale_to=227)

            coords = getXYPinCoords(find_pincoords_line_model, normalized)

            pt1 = round(coords[0]), round(coords[1])
            pt1 = translate(pt1, 227, expand_to, area_start)
            x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)

            components["Unknown"].append({
                "class": "LineEnd",
                "name": f"LE{lineEndAreaIdx}",
                "coord": [x1 + area_start[0], y1 + area_start[1]],
                "pin": pin1
            })

            cv2.putText(canvas_image, f"lineendarea#{lineEndAreaIdx}", (lineareaMinPoint[0], lineareaMinPoint[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(canvas_image, lineareaMinPoint, lineareaMaxPoint, (b, g, r), 10)
            cv2.putText(canvas_image, pin1, (x1 + area_start[0], y1 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            cv2.circle(canvas_image, (x1 + area_start[0], y1 + area_start[1]), 20, (255, 0, 255), cv2.FILLED)
            
    """

    cv2.imwrite("target_img.jpg", target_image)
    cv2.imwrite("canvas_image.jpg", canvas_image)

    # return jsonify({"components": components})
    return {"components": components}


def init():
    global body_pinmap, vol_pinmap, pinmap, pinmap_shape, start_point

    PADDING = 0

    pinmap = json.load(open("static/data/pinmap.json"))
    pinmap_shape = pinmap["shape"]

    body_pinmap = breadboard_bodypin_df(pinmap, PADDING)
    vol_pinmap = breadboard_voltagepin_df(pinmap, PADDING)

    start_point = np.array(
        [
            [PADDING, PADDING],
            [pinmap_shape[1] + PADDING, PADDING],
            [pinmap_shape[1] + PADDING, pinmap_shape[0] + PADDING],
            [PADDING, pinmap_shape[0] + PADDING],
        ],
        dtype=np.float32,
    )


def model_loading():
    global models
    print("Init...")

    with tqdm(total=6) as pbar:
        keys = [
            "resistor.area",
            "resistor.body",
            "line.area",
            "line.endpoint",
            "resistor.pin",
            "line.pin",
        ]

        model_paths = [
            MODEL_RESISTORAREA_PATH,
            MODEL_RESISTORBODY_PATH,
            MODEL_LINEAREA_PATH,
            MODEL_LINEENDAREA_PATH,
            "../model/ResNet152V2.h5",
            "../model/findCoordinLineEnd.h5",
        ]

        for key, model_path in zip(keys[:4], model_paths[:4]):
            print(key, "모델 생성 중")
            start = time.process_time()
            models[key] = torch.hub.load(
                "ultralytics/yolov5", "custom", path=model_path
            )
            end = time.process_time()
            print(key, "모델 생성 완료,", end - start)
            pbar.update(1)

        with tf.device("/cpu:0"):
            for key, model_path in zip(keys[4:], model_paths[4:]):
                print(key, "모델 생성 중")
                start = time.process_time()
                models[key] = tf.keras.models.load_model(model_path)
                end = time.process_time()
                print(key, "모델 생성 완료,", end - start)
                pbar.update(1)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    model_loading()
    app.run(debug=True, host="0.0.0.0", port=3000)
