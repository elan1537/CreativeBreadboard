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

circuit_component_data = []
body_pinmap = None
vol_pinmap = None
pinmap = None
pinmap_shape = None
start_point = None

find_pincoords_resi_model = None
find_pincoords_line_model = None

V = 5
PADDING = 200


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

@app.route("/")
def main():
    return "hi"

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

        # access_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

        # if session.get('visitor') is None:
        #     session['visitor'] = {access_ip: {}}

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

        # 딥러닝 데이터셋 추가 시작
        name = data["img_name"].replace(".jpeg", "").replace(".jpg", "").replace(".JPG" ,"")
        filepath = findfile(f"{name}.json", PROJECT_PATH)
        json.dump(pts, open(f"./static/uploads/check_points/{name}.json", "w"))
        copy(filepath, "/Users/se_park/Library/Mobile Documents/com~apple~CloudDocs/2022 Soongsil/1. CS/CreativeBreadboard/backend/static/uploads/annotation")
        cv2.imwrite(f"./static/uploads/origin_img/{data['img_name']}", target_image)
        # 딥러닝 데이터셋 추가 끝

        _, buffer = cv2.imencode('.jpg', res)
        jpg_as_text = base64.b64encode(buffer).decode()
        res = requests.post("http://localhost:3000/detect", json=json.dumps({'pts': base_point.tolist(), 'img_res': jpg_as_text, 'scale': scale}))
    
        img_data = res.json()

        return jsonify(img_data)

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

    cv2.imwrite("./target_img.jpg", target_image)
    
    get_resistor_area_picking_image, resistor_area_points, resistor_area_pd = checkResistorArea(target_image, pts)
    get_resistor_body_picking_image, resistor_body_points, resistor_body_pd = checkResistorBody(target_image, pts)
    get_linearea_picking_image,      linearea_points,      line_area_pd = checkLinearea(target_image, pts)
    get_lineendarea_picking_image, lineendarea_points, line_endarea_pd = checkLineEndArea(target_image, pts)

    resistor_body_obj = json.loads(resistor_body_points)
    resistor_area_obj = json.loads(resistor_area_points)
    linearea_obj = json.loads(linearea_points)

    base_point = np.array(pts, np.float32)
    base_point = base_point.astype(np.float32)
    transform_mtrx = cv2.getPerspectiveTransform(start_point, base_point)

    initializePinmaps(body_pinmap, vol_pinmap, transform_mtrx)
    time.sleep(0.1)
    search_map = pd.concat([vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1)

    print("find resistor_area")
    components = []

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

        cv2.rectangle(target_image, minPoint, maxPoint, (b, g, r), 10)

        table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)

        normalized = imgNormalizing(area, scale_to=300)
        coords = getXYPinCoords(find_pincoords_resi_model, normalized)

        pt1 = round(coords[0]), round(coords[1])
        pt2 = round(coords[2]), round(coords[3])

        pt1 = translate(pt1, 300, expand_to, area_start)
        pt2 = translate(pt2, 300, expand_to, area_start)

        x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)
        x2, y2, pin2 = getPinCoords(search_map, table_idx, pt2, area_start)

        row = {"name": "Resistor", "value": 300, "start": pin1, "end": pin2}
        components.append(row)

        cv2.putText(target_image, pin1, (x1 + area_start[0], y1 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.circle(target_image, (x1 + area_start[0], y1 + area_start[1]), 20, (0, 0, 255), cv2.FILLED)
        cv2.putText(target_image, pin2, (x2 + area_start[0], y2 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.circle(target_image, (x2 + area_start[0], y2 + area_start[1]), 20, (20, 0, 255), cv2.FILLED)

    cv2.imwrite(f"./resistor_area.jpg", target_image)

    print("find line_endarea")

    table = line_contains_table(line_area_pd, line_endarea_pd)

    print("table:", table)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    for i in range(len(line_area_pd)):
        linearea = line_area_pd.iloc[i]

        minPoint = round(linearea.xmin)-15, round(linearea.ymin)-15
        maxPoint = round(linearea.xmax)+15, round(linearea.ymax)+15

        area_start, area_end, area = area_padding(target_image, minPoint, maxPoint, base_point[0], base_point[2], 0, True)
        cv2.rectangle(target_image, minPoint, maxPoint, (b, g, r), 10)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    line_components = {}
    for i in range(len(line_endarea_pd)):
        endarea = line_endarea_pd.iloc[i] 

        if len(data) == 0:
            continue

        minPoint = round(endarea.xmin)-15, round(endarea.ymin)-15
        maxPoint = round(endarea.xmax)+15, round(endarea.ymax)+15

        # expand_to = max([maxPoint[0] - minPoint[0], maxPoint[1] - minPoint[1]])
        expand_to = 240
        area_start, area_end, area = area_padding(target_image, minPoint, maxPoint, base_point[0], base_point[2], 0, True)

        cv2.rectangle(target_image, area_start, area_end, (b, g, r), 10)

        cv2.imwrite(f"area__{i}.jpg", area)
        table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)

        if len(table_idx) == 0:
            continue

        normalized = imgNormalizing(area, scale_to=227)
        coords = getXYPinCoords(find_pincoords_line_model, normalized)

        if coords is None:
            coords = area.shape[1]/2, area.shape[0]/2


        # for end_area_id in table.keys():
        #     parent = table[end_area_id]
        #     parent_line_area = line_area_pd[line_area_pd.index == parent]
        #     print(parent_line_area)


        pt1 = round(coords[0]), round(coords[1])

        pt1 = translate(pt1, 227, expand_to, area_start)

        x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)
        # endarea_idx = int(line_endarea_pd.index[i])

        # print(endarea_idx, table[endarea_idx])

        cv2.putText(target_image, pin1, (x1 + area_start[0], y1 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
        cv2.circle(target_image, (x1 + area_start[0], y1 + area_start[1]), 20, (255, 0, 255), cv2.FILLED)
       
        cv2.imwrite(f"./linenedarea.jpg", target_image)

    #     import pprint
        
    #     if table.get(endarea_idx) is not None:
    #         parent_line_idx = table[endarea_idx]
    #         parent_line_area_pd = line_area_pd[line_area_pd.index == parent_line_idx]

    #         print(parent_line_idx)
    #         print("asdfasdfasdf", parent_line_area_pd)

    #         # print(parent_line_area_pd['xmin'])

    #         if (x1 >= parent_line_area_pd['xmin'] and x1 <= parent_line_area_pd['xmax']) and (y1 >= parent_line_area_pd['ymin'] and y1 <= parent_line_area_pd['ymax']):
    #             if line_components.get(parent_line_idx) is None:
    #                 line_components[parent_line_idx] = ({"name": "Line", "id": parent_line_idx, "value": 0, "start": pin1, "start_coords": [x1, y1]})
    #             else:
    #                 temp_pinnum = line_components[parent_line_idx]["start"]
    #                 temp_start_coords = line_components[parent_line_idx]["start_coords"]

    #                 print(temp_pinnum, parent_line_idx)

    #                 if (temp_start_coords[0] + temp_start_coords[1]) > (x1 + y1):
    #                     line_components[parent_line_idx]["start"] = pin1
    #                     line_components[parent_line_idx]["start_coords"] = [x1, y1]

    #                     line_components[parent_line_idx]["end"] = temp_pinnum
    #                     line_components[parent_line_idx]["end_coords"] = temp_start_coords
    #                 else:
    #                     line_components[parent_line_idx]["end"] = pin1
    #                     line_components[parent_line_idx]["end_coords"] = [x1, y1]

    # pprint.pprint(line_components)

        # for e in line_components:
        #     components.append(line_components[e])

    # pprint.pprint(components)

    resistor_body_key = list(resistor_body_obj.keys())
    resistor_area_key = list(resistor_area_obj.keys())
    resistor_count = len(resistor_body_obj)


    # Circuit-38.220428
    detected_components_2 = [ # Circuit-38.220428
        {"name": "Line", "start": "F05", "end": "E13", "id": 0, "value": 0},
        {"name": "Line", "start": "E18", "end": "E29", "id": 1, "value": 0},
        {"name": "R1", "value": 100, "start": "V405", "end": "J05", "id": 0},
        {"name": "R2", "value": 100, "start": "C13", "end": "C18", "id": 1},
        {"name": "R3", "value": 100, "start": "A13", "end": "A18", "id": 2},
        {"name": "R4", "value": 100, "start": "B29", "end": "V124", "id": 3},
    ]

    components = pd.DataFrame(detected_components_2)
    circuit = findNetwork(components)

    print(circuit)

    c = []
    c_idx = 0
    prev_layer = 0

    for i in range(len(circuit)):
        row = circuit.iloc[i]

        if "R" not in row['name']:
            continue
        
        print(row['name'])
        if i >= len(circuit):
            break

        d = {
            "name": row['name'],
            "value": row['value']
        }

        now_layer = row.layer

        if now_layer == 0:
            c.append([d])


        print(now_layer, prev_layer)

        if (now_layer == prev_layer):
            if now_layer != 0:
                c[c_idx].append(d)
        else:
            c.append([d])
            c_idx += 1

        prev_layer = now_layer

    # circuit_component_data = [[{"name": f"R{key}", "value": 10}] for key in resistor_body_key]
    circuit_component_data = c

    print("detect -> circuit_component_data", circuit_component_data)

    R_TH, I, NODE_VOL = calcCurrentAndVoltage(V, circuit_component_data)

    print(R_TH, I, NODE_VOL)

    _, buffer = cv2.imencode('.jpg', get_resistor_body_picking_image)

    cv2.imwrite("get_resistor_body_picking_image.jpg", get_resistor_body_picking_image)
    jpg_as_text = base64.b64encode(buffer).decode()

    return jsonify({
        "result_image": jpg_as_text,
        "origin_img": img_res,
        "circuit": base64.b64encode(drawDiagram(V, circuit_component_data)).decode(),
        "area_points": json.loads(resistor_body_points),
        "detected_components": {
            "resistor_area": json.loads(resistor_area_points),
            "resistor_body": json.loads(resistor_body_points),
            "line_area": json.loads(linearea_points),
            "lineend_area": json.loads(lineendarea_points) 
        },
        "circuit_analysis": {
            "r_th": str(R_TH),
            "node_current": str(I),
            "node_voltage": str(NODE_VOL)
        },
        "scale": scale
    })

@app.route('/result', methods=["GET"])
def result():
    test()
    return send_file('./images/precess_image.jpg', mimetype='image/jpg')

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

    if find_pincoords_line_model is None:
        find_pincoords_resi_model = tf.keras.models.load_model("../model/findCoordsResNet50.h5")
    
    if find_pincoords_resi_model is None:
        find_pincoords_line_model = tf.keras.models.load_model("../model/findCoordinLineEnd.h5")

def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)

def line_contains_table(line_area, line_endarea):
    key_table = dict()

    for i in range(len(line_area)):
        area = line_area.iloc[i]

        for j in range(len(line_endarea)):
            endarea = line_endarea.iloc[j]

            # linearea 안에 포함된 lineend를 찾는다.
            if ((area.xmin < endarea.center_x) and (endarea.center_x < area.xmax)) and ((area.ymin < endarea.center_y) and (endarea.center_y < area.ymax)):            
                if key_table.get(j) != None:
                    print(j, "겹침", key_table[j], i)

                    compare_area_1 = line_area.iloc[key_table[j]]
                    compare_area_2 = line_area.iloc[i]

                    d1_key = j
                    d2_key = i

                    key = None

                    d1 = (compare_area_1.loc[['xmin', 'ymin', 'xmax', 'ymax']] - endarea.loc[['xmin', 'ymin', 'xmax', 'ymax']]).sum()
                    d2 = (compare_area_2.loc[['xmin', 'ymin', 'xmax', 'ymax']] - endarea.loc[['xmin', 'ymin', 'xmax', 'ymax']]).sum()

                    if d1 > d2:
                        key = d2_key
                    else:
                        key = d1_key

                    print(j, '->', key)

                    key_table[j] = key

                else:   
                    key_table[j] = i

            # 겹치는게 있다면 그 중 가장 가까운 영역을 찾는다.
    
    return key_table


if __name__ == "__main__":
    init()
    app.run(debug=False, use_reloader=True, host='0.0.0.0', port=3000)