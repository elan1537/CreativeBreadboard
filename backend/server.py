import datetime
import os, json, cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for, session
from flask_cors import CORS
from findColor import test
from findComponents import *
import requests
import base64
from shutil import copy
from diagram import drawDiagram
from calcVoltageAndCurrent import calcCurrentAndVoltage
import tensorflow as tf
import time
import pandas as pd
import pprint
pd.set_option('mode.chained_assignment', None)

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

SAVE_PATH = "./static/uploads"
PROJECT_PATH = "/Users/se_park/Library/Mobile Documents/com~apple~CloudDocs/2022 Soongsil/1. CS/CreativeBreadboard/images/Circuits"

# jpg_original = base64.b64decode(img_data)
# img_arr = np.frombuffer(jpg_original, np.uint8)
# target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

# cv2.imwrite("result_from_post.jpg", target_image)

app = Flask(__name__, static_folder="./static", template_folder="./templates")
app.secret_key = 'f#@&v08@#&*fnvn'
app.permanent_session_lifetime = datetime.timedelta(hours=4)

# app.config.from_object(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

FILE_IMAGE = None

def set_line_component(line_area_pd, line_endarea_pd, table, search_map, base_point, target_image, canvas_image):
        temp = {
            "Line": {},
            "Unknown": []
        }

        for endAreaIdx in table.keys():
            endarea = line_endarea_pd.iloc[endAreaIdx]
            expand_to = 350

            endAreaminPoint =[round(endarea.xmin)-15, round(endarea.ymin)-15]
            endAreamaxPoint = [round(endarea.xmax)+15, round(endarea.ymax)+15]

            # 시작점이 <0이 나올 수가 있음..
            if endAreaminPoint[0] < 0:
                endAreaminPoint[0] = 0
            
            if endAreaminPoint[1] < 0:
                endAreaminPoint[1] = 0
            
            cv2.putText(canvas_image, f"endarea#{endAreaIdx}", (endAreaminPoint[0], endAreaminPoint[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

            area_start, area_end, area = area_padding(target_image, endAreaminPoint, endAreamaxPoint, base_point[0], base_point[2], 0, False)
            
            table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)
            normalized = imgNormalizing(area, scale_to=227)

            coords = getXYPinCoords(find_pincoords_line_model, normalized)

            pt1 = round(coords[0]), round(coords[1])
            pt1 = translate(pt1, 227, expand_to, area_start)
            x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)    

            toLineArea = table[endAreaIdx]
            linearea = line_area_pd.iloc[toLineArea]

            if temp["Line"].get(f"L{toLineArea}") is not None:
                if temp["Line"][f"L{toLineArea}"].get("start"):
                    temp["Line"][f"L{toLineArea}"]["end"] = pin1
                    temp["Line"][f"L{toLineArea}"]["end_coord"] = [x1 + endAreaminPoint[0], y1 + endAreaminPoint[1]]
                    temp["Line"][f"L{toLineArea}"]["end_endAreaStart"] = endAreaminPoint
                    temp["Line"][f"L{toLineArea}"]["end_endAreaEnd"] = endAreamaxPoint

                else:
                    temp["Line"][f"L{toLineArea}"]["start"] = pin1
                    temp["Line"][f"L{toLineArea}"]["start_coord"] = [x1 + endAreaminPoint[0], y1 + endAreaminPoint[1]]
                    temp["Line"][f"L{toLineArea}"]["start_endAreaStart"] = endAreaminPoint
                    temp["Line"][f"L{toLineArea}"]["start_endAreaEnd"] = endAreamaxPoint
 
            elif temp["Line"].get(f"linearea#{toLineArea}") is None:
                line_component = {
                    "class": "Line",
                    "name": f"L{toLineArea}",
                    "areaStart": [round(linearea.xmin), round(linearea.ymin)],
                    "areaEnd": [round(linearea.xmax), round(linearea.ymax)]
                }

                if "V1" in pin1 or "V3" in pin1:
                    line_component["end"] = pin1
                    line_component["end_coord"] = [x1 + endAreaminPoint[0], y1 + endAreaminPoint[1]]
                    line_component["end_endAreaStart"] = endAreaminPoint
                    line_component["end_endAreaEnd"] = endAreamaxPoint

                elif "V2" in pin1 or "V4" in pin1:
                    line_component["start"] = pin1
                    line_component["start_coord"] = [x1 + endAreaminPoint[0], y1 + endAreaminPoint[1]]
                    line_component["start_endAreaStart"] = endAreaminPoint
                    line_component["start_endAreaEnd"] = endAreamaxPoint

                else:
                    line_component["start"] = pin1
                    line_component["start_coord"] = [x1 + endAreaminPoint[0], y1 + endAreaminPoint[1]]
                    line_component["start_endAreaStart"] = endAreaminPoint
                    line_component["start_endAreaEnd"] = endAreamaxPoint

                temp["Line"][f"L{toLineArea}"] = line_component

            else:
                temp["Unknown"].append({
                    "pin": pin1,
                    "coord": [x1 + endAreaminPoint[0], y1 + endAreaminPoint[1]],
                    "areaStart": endAreaminPoint,
                    'areaEnd': endAreamaxPoint
                })


            cv2.putText(canvas_image, pin1, (x1 + area_start[0], y1 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            cv2.circle(canvas_image, (x1 + area_start[0], y1 + area_start[1]), 20, (255, 0, 255), cv2.FILLED)

        return temp

def set_resistor_component(resistor_area_pd, search_map, base_point, target_image, canvas_image):
        temp = {}
        r = int(random.random() * 255)
        g = int(random.random() * 255)
        b = int(random.random() * 255)

        for i in range(len(resistor_area_pd)):
            data = resistor_area_pd.iloc[i]

            if len(data) == 0:
                continue

            minPoint = round(data.xmin)-15, round(data.ymin)-15
            maxPoint = round(data.xmax)+15, round(data.ymax)+15

            expand_to = max([maxPoint[0] - minPoint[0], maxPoint[1] - minPoint[1]])
            area_start, area_end, area = area_padding(target_image, minPoint, maxPoint, base_point[0], base_point[2], expand_to, True)

            cv2.rectangle(canvas_image, minPoint, maxPoint, (b, g, r), 10)

            table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)

            normalized = imgNormalizing(area, scale_to=300)
            coords = getXYPinCoords(find_pincoords_resi_model, normalized)

            pt1 = round(coords[0]), round(coords[1])
            pt2 = round(coords[2]), round(coords[3])

            pt1 = translate(pt1, 300, expand_to, area_start)
            pt2 = translate(pt2, 300, expand_to, area_start)

            x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)
            x2, y2, pin2 = getPinCoords(search_map, table_idx, pt2, area_start)

            temp[f"R{i}"] = {
                "class": "Resistor", 
                "name": f"R{i}", 
                "start": pin1, 
                "end": pin2, 
                "start_coord": [x1 + area_start[0], y1 + area_start[1]], 
                "end_coord": [x2 + area_start[0], y2 + area_start[1]], 
                "value": 100,
                "areaStart": minPoint,
                "areaEnd": maxPoint
            }

            # 이미지에 표시
            cv2.putText(canvas_image, pin1, (x1 + area_start[0], y1 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            cv2.circle(canvas_image, (x1 + area_start[0], y1 + area_start[1]), 20, (0, 0, 255), cv2.FILLED)
            cv2.putText(canvas_image, pin2, (x2 + area_start[0], y2 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            cv2.circle(canvas_image, (x2 + area_start[0], y2 + area_start[1]), 20, (20, 0, 255), cv2.FILLED)
            # 끝

        return temp

@app.route("/")
def main():
    return "hi"

@app.route("/pinmap", methods=['GET'])
def pinmap():
    global body_pinmap, vol_pinmap
    search_map = None
    row = None
    col = None
    search_pin = request.args.get('pin')

    assert search_pin != None
    assert search_pin[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'V']

    if search_pin[0] == 'V':
        assert int(search_pin[2:]) >= 1 and int(search_pin[2:]) <= 25
        row, col = int(search_pin[2:])-1, search_pin[:2]
    else:
        assert int(search_pin[1:]) >= 1 and int(search_pin[1:]) <= 30
        row, col = int(search_pin[1:])-1, search_pin[0]


    if isinstance(body_pinmap, type(None)):
        search_map = pd.read_json(json.load(open("./warpedPinmap.json")))
    else:    
        search_map = pd.concat([vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1)
    
    print(search_map)
    x, y = search_map.xs(row)[col]['x'], search_map.xs(row)[col]['y']

    return jsonify({
        "coord": [x, y]
    })

@app.route("/resistor", methods=['GET', 'POST'])
def resistor():
    global circuit_component_data

    if request.method == 'GET':
        temp = [[{'name': 'R1', 'value': 100}], [{'name': 'R2', 'value': 100}, {'name': 'R3', 'value': 100}], [{'name': 'R4', 'value': 100}]]

        circuit_component_data = temp

        print(circuit_component_data, type(circuit_component_data), type(temp))

        return jsonify({
            "state": "success",
            "data": circuit_component_data
        })

    if request.method == 'POST':
        resistor_value = request.get_json()

        print(resistor_value)

        for r in resistor_value:
            for row in circuit_component_data:
                for col in row:
                    if r['name'] == col['name']:
                        col['value'] = int(r['value'])

        return jsonify({
            "state": "success"
        })

@app.route("/area", methods=["POST"])
def area():
    if request.method == "POST":
        data = request.get_json()

        for key in data.keys():
            circuit_component_data.append([{"name": f"R{key}", "value": 10}])

        return jsonify({
            "state": "success"
        })

@app.route("/draw", methods=["GET"])
def draw():
    if request.method == 'GET':
        print(circuit_component_data)
        image_bytes = drawDiagram(V, circuit_component_data)
        return jsonify({
            "state": "success",
            "circuit": base64.b64encode(image_bytes).decode()
        })

@app.route("/image", methods=['POST'])
def image():
    if request.method == 'POST':
        global FILE_IMAGE, V
        PADDING = 0
        if request.get_json() != None:
            target_image = cv2.imread("../IMG_5633.JPG", cv2.IMREAD_COLOR)
            points = [[93,29],[99,871],[648,865],[648,27]]
            scale = 0.25
            V = 15

        else:
            img_file = request.files['image']
            data = json.load(request.files['data'])

            img_file_bytes = img_file.stream.read()
            img_arr = np.frombuffer(img_file_bytes, np.uint8)
            target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            points = data["points"]
            scale = float(data["scale"])
            V = int(data["voltage"])

        pts = []
        for point in points:
            pts.append([int(point[0] / scale), int(point[1] / scale)])

        base_point, res = toPerspectiveImage(target_image, np.array(pts), PADDING)

        # print(session['visitor'][access_ip].keys())

        # # 딥러닝 데이터셋 추가 시작
        # name = data["img_name"].replace(".jpeg", "").replace(".jpg", "").replace(".JPG" ,"")
        # filepath = findfile(f"{name}.json", PROJECT_PATH)
        # json.dump(pts, open(f"./static/uploads/check_points/{name}.json", "w"))
        # copy(filepath, "/Users/se_park/Library/Mobile Documents/com~apple~CloudDocs/2022 Soongsil/1. CS/CreativeBreadboard/backend/static/uploads/annotation")
        # cv2.imwrite(f"./static/uploads/origin_img/{data['img_name']}", target_image)
        # # 딥러닝 데이터셋 추가 끝

        _, buffer = cv2.imencode('.jpg', res)
        jpg_as_text = base64.b64encode(buffer).decode()
        res = requests.post("http://137.184.95.69:3000/detect", json=json.dumps({'pts': base_point.tolist(), 'img_res': jpg_as_text, 'scale': scale}))
    
        components = res.json()
        print(type(components))

        result_data = {
            "transformedImg": jpg_as_text,
            "basePoint": base_point.tolist(),
            "voltage": V,
            "scale": 0.25,
            "components": components["components"]
        }

        return jsonify(result_data)

@app.route("/points", methods=['POST'])
def points():
    data = json.load(request.files['data'])
    points = data["points"]
    img_name = data["img_name"]
    print(img_name, points)

    return jsonify({"message": "success"})

@app.route("/calc", methods=["get"])
def calc():
    global circuit_component_data
    R_TH, I, NODE_VOL = calcCurrentAndVoltage(V, circuit_component_data)
    
    return jsonify({
        "circuit_analysis": {
            "r_th": str(R_TH),
            "node_current": str(I),
            "node_voltage": str(NODE_VOL)
        }
    })

@app.route("/network", methods=['GET', 'POST'])
def network():
    if request.method == 'POST':
        global circuit_component_data
        components = request.get_json()

        lines = pd.DataFrame(components["Line"])
        resistors = pd.DataFrame(components["Resistor"])

        components = pd.concat([lines, resistors], axis=1).transpose()

        print(components)

        '''
            일단 지금은 사용자가 모든 핀을 정상적으로 오류를 캐치했을 때를 가정하고 네트워크를 구성하고 있음
            추가적으로 네트워크를 찾다가 잘못된 부분을 alert 하는거 구현해야함.
        '''
        '''
            OMG... D7과 H7이 같은 노드라고 나옴.. 이거 분리..~
        '''

        circuit = findNetwork(components)


        # 아래의 코드는 위에서 계층을 찾으면 그걸 토대로 저항소자만 빼오는 구조
        # 위 데이터의 결과는 

        print(circuit)

        table = {}

        for i in range(len(circuit)):
            row = circuit.iloc[i]

            if "R" not in row['name']:
                continue

            d = {
                "name": row['name'],
                "value": row['value']
            }

            if table.get(row.layer) is None:
                table[int(row.layer)] = [d]
            else:
                table[int(row.layer)].append(d)

        table = [value for _, value in table.items()]

        circuit_component_data = table

        return jsonify({
            "network": table
        })


@app.route("/detect", methods=['POST'])
def detect():
    # 이미지 프로세싱
    global circuit_component_data, vol_pinmap, body_pinmap, start_point

    init()

    data = json.loads(request.get_json())
    pts = data['pts']
    img_res = data['img_res']
    scale = data['scale']
    jpg_original = base64.b64decode(img_res)
    img_arr = np.frombuffer(jpg_original, np.uint8)
    target_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    canvas_image = target_image.copy()

    cv2.imwrite("./target_img.jpg", canvas_image)
    
    _, resistor_area_points, resistor_area_pd = checkResistorArea(target_image, pts)
    _, resistor_body_points, resistor_body_pd = checkResistorBody(target_image, pts)
    _,      linearea_points,      line_area_pd = checkLinearea(target_image, pts)
    _,   lineendarea_points, line_endarea_pd = checkLineEndArea(target_image, pts)

    base_point = np.array(pts, np.float32)
    base_point = base_point.astype(np.float32)
    transform_mtrx = cv2.getPerspectiveTransform(start_point, base_point)

    initializePinmaps(body_pinmap, vol_pinmap, transform_mtrx)
    search_map = pd.concat([vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1)

    json.dump(search_map.to_csv(), open("./warpedPinmap.csv", "w"))

    print("resistor_area_pd:", len(resistor_area_pd))
    print("resistor_body_pd:", len(resistor_body_pd))
    print("line_area_pd:", len(line_area_pd))
    print("line_endarea_pd:", len(line_endarea_pd))

    components = {
        "Resistor": [],
        "Unknown": []
    }

    detected_resistor_components = set_resistor_component(resistor_area_pd, search_map, base_point, target_image, canvas_image)
    components['Resistor'] = detected_resistor_components

    cv2.imwrite("canvas_image.jpg", canvas_image)

    if len(line_area_pd) > 0:
        table = line_contains_table(line_area_pd, line_endarea_pd)

        detected_line_components = set_line_component(line_area_pd, line_endarea_pd, table, search_map, base_point, target_image, canvas_image)
        components["Line"] = detected_line_components["Line"]
        components["Unknown"] = detected_line_components["Unknown"]

        for lineAreaIdx in table.values():
            r = int(random.random() * 255)
            g = int(random.random() * 255)
            b = int(random.random() * 255)

            linearea = line_area_pd.iloc[lineAreaIdx]

            lineareaMinPoint = round(linearea.xmin) - 30, round(linearea.ymin) - 30
            lineareaMaxPoint = round(linearea.xmax) + 30, round(linearea.ymax) + 30
        
            cv2.putText(canvas_image, f"linearea#{lineAreaIdx}", (lineareaMinPoint[0], lineareaMinPoint[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(canvas_image, lineareaMinPoint, lineareaMaxPoint, (b, g, r), 10)

    elif len(line_endarea_pd) > 0:
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

        
    cv2.imwrite("target_img.jpg", target_image)
    cv2.imwrite("canvas_image.jpg", canvas_image)

    return jsonify({"components": components})

@app.route('/result', methods=["GET"])
def result():
    test()
    return send_file('./images/precess_image.jpg', mimetype='image/jpg')

@app.route("/warpedImg", methods=["GET"])
def warpedImg():
    img = cv2.imread("./target_img.jpg", cv2.IMREAD_COLOR)
    _, buffer = cv2.imencode('.jpg', img)
    img = base64.b64encode(buffer).decode()
    return jsonify({
            "state": "success",
            "img": img
        })

def findNetwork(components):
    circuits = pd.DataFrame()
    layer_count = 0

    start_comp = components[components.start.str.contains("V")]

    if len(start_comp) >= 2: # 시작이 만약 2개 이상일 때
        components.drop(index=start_comp.index, inplace=True)
        findNetwork(components)
    else:
        components.drop(index=start_comp.index, inplace=True)

        circuits = pd.concat([circuits, start_comp])
        circuits['layer'] = layer_count

        next_point = start_comp
        layer_count += 1

        while not components.empty:
            next_point = next_point.end.to_string(index=False)
            row = next_point[0]
            col = next_point[1:]

            next_point = components[components['start'].str.contains(f"{col}")]

            components.drop(index=next_point.index, inplace=True)

            if next_point.empty:
                next_point = components[components['start'].str.contains(f"{row}")]

            next_point['layer'] = layer_count
            circuits = pd.concat([circuits, next_point], axis=0)

            if len(next_point) > 1:
                next_point = next_point.iloc[[0]]
                # -> 또 검색 해야함.. 
                # 만약 병렬된 곳에서 타고타고 들어갈 수도..

            layer_count += 1


    return circuits

def init():
    global                          \
        body_pinmap,                \
        vol_pinmap,                 \
        pinmap,                     \
        pinmap_shape,               \
        start_point,                \
        find_pincoords_resi_model,  \
        find_pincoords_line_model
    
    PADDING = 0

    pinmap = json.load(open("static/data/pinmap.json"))
    pinmap_shape = pinmap["shape"]

    body_pinmap = breadboard_bodypin_df(pinmap, PADDING)
    vol_pinmap = breadboard_voltagepin_df(pinmap, PADDING)

    start_point = np.array([
        [PADDING, PADDING], 
        [pinmap_shape[1] + PADDING, PADDING], 
        [pinmap_shape[1] + PADDING, pinmap_shape[0] + PADDING],  
        [PADDING, pinmap_shape[0] + PADDING]
    ], dtype=np.float32)

    if find_pincoords_resi_model is None:
        print("resi 모델 생성")
        find_pincoords_resi_model = tf.keras.models.load_model("../model/ResNet152V2.h5")
    
    if find_pincoords_line_model is None:
        print("line 모델 생성")
        find_pincoords_line_model = tf.keras.models.load_model("../model/findCoordinLineEnd.h5")

def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)

def line_contains_table(line_area, line_endarea):
    key_table = dict()

    print("line_area", len(line_area))
    print("line_endarea", len(line_endarea))

    for i in range(len(line_area)):
        area = line_area.iloc[i]

        for j in range(len(line_endarea)):
            endarea = line_endarea.iloc[j]

            # print(f"전선영역{i}와 전선연결부{j}와 비교 중")

            # print(f"{area.xmin} < {endarea.center_x} < {area.xmax} || {area.ymin} < {endarea.center_y} < {area.ymax}")

            # linearea 안에 포함된 lineend를 찾는다.
            if ((area.xmin < endarea.center_x) and (endarea.center_x < area.xmax)) and ((area.ymin < endarea.center_y) and (endarea.center_y < area.ymax)):            
                if key_table.get(j) != None:
                    # compareKey(key_table, i, key_table[j], line_area, endarea) # >> 키값 전달이 어디서 꼬이는 듯

                    newArea = line_area.iloc[i]
                    oldArea = line_area.iloc[key_table[j]]
                    end_area = endarea

                    d1 = ((newArea.loc[['xmax', 'ymax']] - end_area.loc[['xmin', 'ymin']]) - (end_area.loc[['xmin', 'ymin']] - newArea.loc[['xmin', 'ymin']])).sum()
                    d2 = ((oldArea.loc[['xmax', 'ymax']] - end_area.loc[['xmin', 'ymin']]) - (end_area.loc[['xmin', 'ymin']] - oldArea.loc[['xmin', 'ymin']])).sum()

                    newAreaMin = newArea.loc['xmin'], newArea.loc['ymin']
                    newAreaMax = newArea.loc['xmax'], newArea.loc['ymax']
                    oldAreaMin = oldArea.loc['xmin'], oldArea.loc['ymin']
                    oldAreaMax = oldArea.loc['xmax'], oldArea.loc['ymin']
                    endAreaMin = end_area.loc['xmin'], end_area.loc['ymin']
                    endAreaMax = end_area.loc['xmax'], end_area.loc['ymax']

                    d1 = (endAreaMin[0] - newAreaMin[0]) + (endAreaMin[1] - newAreaMin[1]) + (newAreaMax[0] - endAreaMax[0]) + (newAreaMax[1] - endAreaMax[1])
                    d2 = (endAreaMin[0] - oldAreaMin[0]) + (endAreaMin[1] - oldAreaMin[1]) + (oldAreaMax[0] - endAreaMax[0]) + (oldAreaMax[1] - endAreaMax[1])
                    # d1과 d2이 음수가 나오는 의미가..?

                    if d1 > d2:
                        key = i
                        
                    elif d1 < d2:
                        key = key_table[j]

                    if key_table.get(j) != None:
                        # tempKey = key_table[key]
                        key_table[j] = key
                        
                    else:
                        key_table[j] = key

                else:   
                    key_table[j] = i
    
    return key_table


if __name__ == "__main__":
    app.run(debug=False, use_reloader=True, host='0.0.0.0', port=3000)