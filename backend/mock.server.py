import datetime
import os, json, cv2
import requests
import base64
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import io
import argparse

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from findComponents import toPerspectiveImage, findNetwork
from data_processing.diagram import drawDiagram
from data_processing.calcVoltageAndCurrent import calcCurrentAndVoltage

pd.set_option("mode.chained_assignment", None)

app = Flask(__name__, static_folder="./static")
app.secret_key = "f#@&v08@#&*fnvn"
app.permanent_session_lifetime = datetime.timedelta(hours=4)


V = None
vol_pinmap = None
body_pinmap = None
circuit_component_data = None

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def main():
    """
    서버 동작 체크 여부
    """
    return "Hello, I'm on"


@app.route("/image/<data_type>", methods=["POST"])
def image(data_type: int):
    if request.method == "POST":
        try:
            global V, body_pinmap, vol_pinmap, circuit_component_data
            PADDING = 200
            target_image = None
            points = None
            scale = None
            dummy = None

            print(data_type, request.files["circuitImage"], request.form["points"])

            if request.files["circuitImage"]:
                pass
            else:
                raise Exception("회로 이미지를 전송하지 않음")

            if request.form["points"]:
                pass
            else:
                raise Exception("이미지 필수 정보 (4 꼭짓점, 전압, 스케일)가 전송되지 않음")

            dummy_id = int(data_type)

            with open("./mock_data/conponents.mock.json", "r") as f:
                dummy = json.load(f)

            dummy_key = list(dummy.keys())[dummy_id]
            points = dummy[dummy_key]["meta"]["points"]
            scale = float(dummy[dummy_key]["meta"]["scale"])
            V = int(dummy[dummy_key]["meta"]["voltage"])
            components = dummy[dummy_key]["components"]

            search_map = pd.read_pickle(
                f"./mock_data/pinmap/{request.files['circuitImage'].filename}.pkl"
            )

            vol_pinmap = pd.concat(
                [search_map.iloc[:, 0:4], search_map.iloc[:, 24:28]], axis=1
            )

            body_pinmap = search_map.iloc[:, 4:24]

            target_image = cv2.imdecode(
                np.frombuffer(request.files["circuitImage"].read(), np.uint8),
                cv2.IMREAD_COLOR,
            )

            pts = []
            for point in points:
                pts.append([int(point[0] / scale), int(point[1] / scale)])

            base_point, target_image = toPerspectiveImage(
                target_image, np.array(pts), PADDING
            )

            _, buffer = cv2.imencode(".jpg", target_image)
            transformedImg_base64 = base64.b64encode(buffer).decode()

            result_data = {
                "transformedImg": transformedImg_base64,
                "basePoint": base_point.tolist(),
                "voltage": V,
                "scale": 0.25,
                "components": components,
            }

            return jsonify(result_data)

        except Exception as e:
            return jsonify({"message": str(e)})


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

        if isinstance(circuit, dict):
            return jsonify({"message": circuit["message"]})

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

    if request.method == "GET":
        if circuit_component_data is not None:
            return jsonify({"network": circuit_component_data})
        else:
            return jsonify({"message": "회로 이미지를 먼저 업로드하세요"})


@app.route("/pinmap", methods=["GET"])
def pinmap():
    global body_pinmap, vol_pinmap
    search_map = None
    row = None
    col = None
    search_pin = request.args.get("pin")

    try:
        assert search_pin != None
        assert search_pin[0] in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "V"]

        search_map = pd.concat(
            [vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1
        )

        if search_pin[0] == "V":
            assert int(search_pin[2:]) >= 1 and int(search_pin[2:]) <= 25
            row, col = int(search_pin[2:]) - 1, search_pin[:2]
        else:
            assert int(search_pin[1:]) >= 1 and int(search_pin[1:]) <= 30
            row, col = int(search_pin[1:]) - 1, search_pin[0]

        x, y = search_map.xs(row)[col]["x"], search_map.xs(row)[col]["y"]

        return jsonify({"coord": [x, y]})

    except AttributeError as e:
        return jsonify({"message": "회로 이미지를 먼저 업로드하세요"})

    except AssertionError:
        return jsonify({"message": "유효한 브레드보드 핀 범위가 아님. (V11(V101)~V425, A1(A01)~J25)"})


@app.route("/draw", methods=["GET"])
def draw():
    if request.method == "GET":
        global circuit_component_data
        try:
            image_bytes = drawDiagram(V, circuit_component_data)
            # return jsonify(
            #     {"state": "success", "circuit": base64.b64encode(image_bytes).decode()}
            # )
            return send_file(
                io.BytesIO(image_bytes),
                mimetype="image/jpeg",
                attachment_filename="circuitDiagram.jpeg",
            )
        except Exception:
            return jsonify({"message": "회로 이미지를 먼저 업로드하세요"})


@app.route("/calc", methods=["get"])
def calc():
    """
    분석된 회로에서의 이론적인 노드 전압과 출력 전류 그리고 합성저항값을 반환받는다.
    """
    global circuit_component_data

    if circuit_component_data is None:
        return jsonify({"message": "회로 사진을 먼저 업로드하세요"})

    if V is None:
        return jsonify({"message": "전압값이 지정되어있지 않음"})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="서버 시작")
    parser.add_argument("--debug", default=False)
    parser.add_argument("--port", default=7080)
    args = parser.parse_args()

    app.run(debug=args.debug, host="0.0.0.0", port=args.port)
