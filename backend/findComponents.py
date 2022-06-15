import random
import cv2
import numpy as np
import torch
import pandas as pd

MODEL_RESISTORAREA_PATH = "../model/resistor-area.model.pt"
MODEL_RESISTORBODY_PATH = "../model/resistor.body.pt"
MODEL_LINEAREA_PATH = "../model/line-area.model.pt"
MODEL_LINEENDAREA_PATH = "../model/line-endpoint.model.pt"

''' 
    이미지 왜곡 수정 함수 
    padding 만큼 여백을 주어 왜곡 수정을 한다.
'''

resistor_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_RESISTORAREA_PATH)
linearea_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEAREA_PATH)
lineendarea_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEENDAREA_PATH)

def toPerspectiveImage(img, points, padding = 0):
    if points.ndim != 2:
        points = points.reshape((-1, 2))
    
    sm = points.sum(axis = 1)
    diff = np.diff(points, axis=1)

    topLeft = points[np.argmin(sm)]
    bottomRight = points[np.argmax(sm)]
    topRight = points[np.argmin(diff)]
    bottomLeft = points[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    width = max([w1, w2])
    height = max([h1, h2])

    pts2 = np.float32([
        [padding, padding],
        [width - 1 + padding, padding],
        [width - 1 + padding, height - 1 + padding],
        [padding, height - 1 + padding]
    ])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    return pts2, cv2.warpPerspective(img, mtrx, (width + 2*padding, height + 2*padding), flags=cv2.INTER_CUBIC)

def area_padding(old_area, from_: tuple, to_: tuple, canvas_start: tuple or list, canvas_to: tuple or list, expand_to = 0, blank = False):
    '''
        타겟 영역에서 직사각형 영역 from_에서 to_ 까지 crop한다.
        padding이 이뤄진 전체 영역인 canvas_start, canvas_to를 가지고
        새롭게 crop 하는영역이 관심영역 (브레드보드 영역 안쪽)에만 잘리게 한다.

        expand_to로 중점을 중심으로 주위 핀을 찾기위해 확장한다.
    '''

    # 범위를 넘어가나?
    x_ = [from_[0], to_[0]]
    y_ = [from_[1], to_[1]]

    if from_[0] > to_[0]: # 오른쪽으로 범위가 넘어감
        # y_ = [canvas_start[1], canvas_to[1]]
        x_ = [canvas_start[0], to_[0]]

    if to_[0] < from_[0]: # 왼쪽으로 범위가 넘어감
        # y_ = canvas_start[1], canvas_to[3]
        x_ = [from_[0], canvas_to[0]]

    if to_[1] < from_[1]: # 위쪽으로 범위가 넘어감
        y_ = [from_[1], canvas_to[1]]
        # x_ = canvas_start[0], canvas_to[0]

    if from_[1] > to_[1]: # 아래쪽으로 범위가 넘어감
        y_ = [canvas_start[1], to_[1]]
        # x_ = canvas_start[0], canvas_to[0]

    to_area = old_area[y_[0]:y_[1], x_[0]:x_[1]]

    c_x = to_area.shape[1]/2
    c_y = to_area.shape[0]/2

    # 110, 154 -> 350, 350
    # (350 - 110)/2 = 120, (350-154)/2 = 98

    if expand_to != 0:
        add_width  = int((expand_to/2 - c_x))
        add_height = int((expand_to/2 - c_y))

        x_[0] -= add_width
        y_[0] -= add_height
        x_[1] += add_width
        y_[1] += add_height

        '''
            지금 이 부분에서 문제가 있는 듯함 -> 일부 사진에서 포인트가 안맞음
        '''
        # padding으로 확대했는데 범위를 넘어가면..
        # 범위를 넘어간 만큼 가능한 공간에서 다시 재확장된다.
        a = 0
        b = 0
        c = 0
        d = 0

        if x_[1] > canvas_to[0]:
            x_[1] -= add_width
            x_[0] -= add_width
            a -= add_width
            b += add_width
            print(" xmax bound")

        if x_[0] < canvas_start[0]:
            x_[0] += add_width
            x_[1] += add_width
            a += add_width
            b -= add_width
            print(" xmin bound")

        if y_[1] > canvas_to[1]:
            y_[1] -= add_height
            y_[0] -= add_height
            c -= add_height
            d += add_height
            print(" ymax bound")
        
        if y_[0] < canvas_start[1]:
            y_[1] += add_height
            y_[0] += add_height 
            c += add_height
            d -= add_height
            print(" ymin bound")

        if blank:
            canvas = cv2.copyMakeBorder(to_area, add_height-c, add_height-d, add_width-a, add_width-b, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            print(add_height-c, add_height-d, add_width-a, add_width-b)

            return (x_[0], y_[0]), (x_[1], y_[1]), canvas
        else:
            # padding 만큼 확장된 결과
            expanded = old_area[y_[0]:y_[1], x_[0]:x_[1]]
            return (x_[0], y_[0]), (x_[1], y_[1]), expanded
    else:
        return (x_[0], y_[0]), (x_[1], y_[1]), to_area

def rectArea(df):
    return (df.xmax - df.xmin) * (df.ymax - df.ymin)

def processDataFrame(origin_data, column_name, confidence=0.5):
    df = origin_data[(origin_data["name"] == column_name) & (origin_data["confidence"] > confidence)].copy()
    df['area']      = df.apply(rectArea, axis=1)
    df['center_x']  = df.apply(lambda row: int((row.xmax + row.xmin) / 2), axis=1)
    df['center_y']  = df.apply(lambda row: int((row.ymax + row.ymin) / 2), axis=1)
    df['length']    = df.apply(lambda row: int((row.xmax - row.xmin)), axis=1)
    df['width']     = df.apply(lambda row: int((row.ymax - row.ymin)), axis=1)
    df['distance_from_origin'] = df.apply(lambda row: int((row.xmin + row.ymin)), axis=1)
    df = df.sort_values(by=['distance_from_origin'], ascending=True)

    return df

def checkLinearea(target, base_point):
    global linearea_detect_model

    if linearea_detect_model is None:
        linearea_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEAREA_PATH)

    detect_area = pd.DataFrame(linearea_detect_model(target).pandas().xyxy[0])

    if detect_area.empty:
        return target, "{}"

    line_area = processDataFrame(detect_area, "line-area", 0.3)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    for i in range(len(line_area)):
        data = line_area.iloc[i]

        result = None
        
        p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

        cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (b, g, r), 10)

        resistor_area_from, resistor_area_to, area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], blank=False)

        result = area.copy()

        area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        _, area = cv2.threshold(area, 
        -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=4)
        cv2.rectangle(result, (0, 0), (p[2] - p[0], p[3] - p[1]), (0, 255, 0), 3)
    
    cv2.imwrite(f"line_total_result.jpg", target)
    return target, line_area.transpose().to_json()

def checkLineEndArea(target, base_point):
    global lineendarea_detect_model

    if lineendarea_detect_model is None:
        lineendarea_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEAREA_PATH)

    detect_area = pd.DataFrame(lineendarea_detect_model(target).pandas().xyxy[0])
    line_end_area = processDataFrame(detect_area, "line-endpoint", 0.3)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    for i in range(len(line_end_area)):
        data = line_end_area.iloc[i]

        result = None
        
        p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

        cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (b, g, r), 10)

        resistor_area_from, resistor_area_to, area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], blank=False)

        result = area.copy()

        area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=4)
        cv2.rectangle(result, (0, 0), (p[2] - p[0], p[3] - p[1]), (0, 255, 0), 3)

    cv2.imwrite(f"linearea_total_result.jpg", target)
    return target, line_end_area.transpose().to_json()

def checkResistorArea(target, base_point):
    ''' Resistor DataFrame 처리 '''
    global resistor_detect_model

    if resistor_detect_model is None:
        resistor_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_RESISTORAREA_PATH)

    detect_area = pd.DataFrame(resistor_detect_model(target).pandas().xyxy[0])

    resistor_area = processDataFrame(detect_area, "resistor-area", 0.7)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    for i in range(len(resistor_area)):
        data = resistor_area.iloc[i]

        result = None
        
        p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

        cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (b, g, r), 10)

        resistor_area_from, resistor_area_to, area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], blank=False)

        result = area.copy()

        area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=4)
        cv2.rectangle(result, (0, 0), (p[2] - p[0], p[3] - p[1]), (0, 255, 0), 3)
                
    cv2.imwrite(f"total_result.jpg", target)
    return target, resistor_area.transpose().to_json()

def checkResistorBody(target, base_point):
    ''' Resistor DataFrame 처리 '''
    global resistor_detect_model

    if resistor_detect_model is None:
        resistor_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_RESISTORAREA_PATH)

    detect_area = pd.DataFrame(resistor_detect_model(target).pandas().xyxy[0])

    resistor_body = processDataFrame(detect_area, "resistor-body", 0.7)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)


    for i in range(len(resistor_body)):
        data = resistor_body.iloc[i]

        if data.confidence > 0.5:
            p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

            area_from, area_to, area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2])
            
            cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (10, 10, 210), 10)

            result = area.copy()

            area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

            area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=4)
            cv2.rectangle(result, (0, 0), (p[2] - p[0], p[3] - p[1]), (0, 255, 0), 3)  

    
    cv2.imwrite(f"total_result.jpg", target)
    return target, resistor_body.transpose().to_json()