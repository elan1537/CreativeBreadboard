import random
import cv2
import numpy as np
import torch
import pandas as pd
import math

MODEL_RESISTORAREA_PATH = "../model/resistor-area.model.pt"
MODEL_RESISTORBODY_PATH = "../model/resistor.body.pt"
MODEL_LINEAREA_PATH = "../model/line-area.model.pt"
MODEL_LINEENDAREA_PATH = "../model/line-endpoint.model.pt"

BREADBOARD_COL_INDEX = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'F', 'F', 'G', 'G', 'H', 'H', 'I', 'I', 'J', 'J']
BREADBOARD_ROW_INDEX = range(30)

PADDING = 150

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
            # print(" xmax bound")

        if x_[0] < canvas_start[0]:
            x_[0] += add_width
            x_[1] += add_width
            a += add_width
            b -= add_width
            # print(" xmin bound")

        if y_[1] > canvas_to[1]:
            y_[1] -= add_height
            y_[0] -= add_height
            c -= add_height
            d += add_height
            # print(" ymax bound")
        
        if y_[0] < canvas_start[1]:
            y_[1] += add_height
            y_[0] += add_height 
            c += add_height
            d -= add_height
            # print(" ymin bound")

        if blank:
            canvas = cv2.copyMakeBorder(to_area, add_height-c, add_height-d, add_width-a, add_width-b, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # print(add_height-c, add_height-d, add_width-a, add_width-b)

            return (x_[0], y_[0]), (x_[1], y_[1]), canvas
        else:
            # padding 만큼 확장된 결과
            expanded = old_area[y_[0]:y_[1], x_[0]:x_[1]]
            return (x_[0], y_[0]), (x_[1], y_[1]), expanded
    else:
        return (x_[0], y_[0]), (x_[1], y_[1]), to_area

def rectArea(df):
    return (df.xmax - df.xmin) * (df.ymax - df.ymin)

def processDataFrame(origin_data, column_name, confidence=0.1):
    df = origin_data[(origin_data["name"] == column_name) & (origin_data["confidence"] > confidence)].copy()

    if len(df) > 0:
        df['area']      = df.apply(rectArea, axis=1)
        df['center_x']  = df.apply(lambda row: int((row.xmax + row.xmin) / 2), axis=1)
        df['center_y']  = df.apply(lambda row: int((row.ymax + row.ymin) / 2), axis=1)
        df['length']    = df.apply(lambda row: int((row.xmax - row.xmin)), axis=1)
        df['width']     = df.apply(lambda row: int((row.ymax - row.ymin)), axis=1)
        df['distance_from_origin'] = df.apply(lambda row: int((row.xmin + row.ymin)), axis=1)
        df = df.sort_values(by=['distance_from_origin'], ascending=True)

        return df
    else:
        return pd.DataFrame({})

def checkLinearea(target, base_point):
    global linearea_detect_model

    if linearea_detect_model is None:
        linearea_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEAREA_PATH)

    detect_area = pd.DataFrame(linearea_detect_model(target).pandas().xyxy[0])

    print("checkLinearea", len(detect_area))

    if len(detect_area) > 0:
        line_area = processDataFrame(detect_area, "line-area", 0.5)
    else:
        line_area = {}

    return target, line_area.transpose().to_json(), line_area

def checkLineEndArea(target, base_point):
    global lineendarea_detect_model

    if lineendarea_detect_model is None:
        lineendarea_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEAREA_PATH)

    detect_area = pd.DataFrame(lineendarea_detect_model(target).pandas().xyxy[0])

    print("checkLineEndArea", len(detect_area))

    line_end_area = processDataFrame(detect_area, "line-endpoint", 0.5)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    # for i in range(len(line_end_area)):
    #     data = line_end_area.iloc[i]

    #     result = None
        
    #     p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

    #     cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (b, g, r), 10)

    #     resistor_area_from, resistor_area_to, area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], blank=True)

    #     result = area.copy()

    #     area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    #     _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    #     area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=4)
    #     cv2.rectangle(result, (0, 0), (p[2] - p[0], p[3] - p[1]), (0, 255, 0), 3)

    cv2.imwrite(f"linearea_total_result.jpg", target)

    return target, line_end_area.transpose().to_json(), line_end_area

def checkResistorArea(target, base_point):
    ''' Resistor DataFrame 처리 '''
    global resistor_detect_model

    if resistor_detect_model is None:
        resistor_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_RESISTORAREA_PATH)

    detect_area = pd.DataFrame(resistor_detect_model(target).pandas().xyxy[0])

    print("checkResistorArea", len(detect_area))

    resistor_area = processDataFrame(detect_area, "resistor-area", 0.5)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)

    # for i in range(len(resistor_area)):
    #     data = resistor_area.iloc[i]

    #     result = None
        
    #     p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

    #     cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (b, g, r), 10)

    #     resistor_area_from, resistor_area_to, area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], blank=False)

    #     result = area.copy()

    #     area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    #     _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    #     area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=4)
    #     cv2.rectangle(result, (0, 0), (p[2] - p[0], p[3] - p[1]), (0, 255, 0), 3)
                
    # cv2.imwrite(f"total_result.jpg", target)
    return target, resistor_area.transpose().to_json(), resistor_area

def checkResistorBody(target, base_point):
    ''' Resistor DataFrame 처리 '''
    global resistor_detect_model

    if resistor_detect_model is None:
        resistor_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_RESISTORAREA_PATH)

    detect_area = pd.DataFrame(resistor_detect_model(target).pandas().xyxy[0])

    print("checkResistorBody", len(detect_area))

    resistor_body = processDataFrame(detect_area, "resistor-body", 0.5)

    r = int(random.random() * 255)
    g = int(random.random() * 255)
    b = int(random.random() * 255)


    # for i in range(len(resistor_body)):
    #     data = resistor_body.iloc[i]

    #     if data.confidence > 0.5:
    #         p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

    #         area_from, area_to, area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2])
            
    #         cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (10, 10, 210), 10)

    #         result = area.copy()

    #         area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    #         _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    #         kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    #         area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=4)
    #         cv2.rectangle(result, (0, 0), (p[2] - p[0], p[3] - p[1]), (0, 255, 0), 3)  

    
    # cv2.imwrite(f"total_result.jpg", target)
    return target, resistor_body.transpose().to_json(), resistor_body

def findCandidateCoords(area_start, area_end, bodymap, volmap):
    global PADDING
    search_map = None
    table_idx = []

    index_map = {
        0: "V1",
        1: "V2",
        2: "A",
        3: "B",
        4: "C",
        5: "D",
        6: "E",
        7: "F",
        8: "G",
        9: "H",
        10: "I",
        11: "J",
        12: "V3",
        13: "V4"
    }

    search_map = pd.concat([volmap.iloc[:, 0:4], bodymap, volmap.iloc[:, 4:8]], axis=1)

    pin_x = [search_map.iloc[:, col].mean() - PADDING for col in range(0, len(search_map.columns), 2)]
    pin_x = np.array(pin_x, np.uint32)
    range_x = np.array(np.where(((pin_x >= area_start[0]) & (pin_x <= area_end[0]))))[0].tolist()
    col_name = np.array([index_map[idx] for idx in range_x], np.str0)

    isVContains = np.array(["V" in col for col in col_name], np.bool_)

    vol_map = col_name[isVContains == True]
    pin_map = col_name[isVContains == False]

    for col in pin_map:
        pin_y = [search_map.iloc[row, range(5, 24, 2)].mean() - PADDING for row in range(30)]

        pin_y = np.array(pin_y, np.float64)
        range_y = np.array(np.where(((pin_y >= area_start[1]) & (pin_y <= area_end[1]))))[0].tolist()
        row_name = [idx+1 for idx in range_y]

        for row in row_name:
            table_idx.append(f"{col}{row}")


    for col in vol_map:
        pin_y = [search_map.iloc[row, [1, 3, 25, 27]].mean() - PADDING for row in range(25)]

        pin_y = np.array(pin_y, np.float64)
        range_y = np.array(np.where(((pin_y >= area_start[1]) & (pin_y <= area_end[1]))))[0].tolist()
        row_name = [idx+1 for idx in range_y]

        for row in row_name:
            table_idx.append(f"{col}{row}")
    
    return table_idx

def xyToHomocoords(x, y):
    return np.array([
        [x],
        [y],
        [1]
    ])
            
def homocoordsToxy(v):
    return int(v[0][0]/v[0][2]), int(v[0][1]/v[0][2])

def imgNormalizing(src, scale_to):
    print(src.shape)
    img_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img = cv2.resize(img_output, (scale_to, scale_to))
    img = img.reshape(-1, scale_to, scale_to, 3)

    mean = np.mean(img, axis=(0, 1, 2, 3))
    std = np.std(img, axis=(0, 1, 2, 3))

    img = (img-mean)/(std + 1e-5)
    return img

def getXYPinCoords(model, src):
    if model:
        c = list(model.predict(src)[0])
        return c
    else:
        print("nomodel")
        return [0, 0]

def getPinCoords(search_map, candidates, coord, area_start):
    global PADDING

    distance = 1000000
    final_coord = [0, 0]
    pin_name = -1

    for candidate in candidates:
        if "V" in candidate:
            col = str(candidate[:2])
            row = int(candidate[2:]) - 1
        else:
            col = str(candidate[:1])
            row = int(candidate[1:]) - 1
        
        c_x = (search_map.xs(row)[col]['x'] - area_start[0] - PADDING)
        c_y = (search_map.xs(row)[col]['y'] - area_start[1] - PADDING)

        n_x = coord[0] - area_start[0]
        n_y = coord[1] - area_start[1]

        d = math.sqrt((c_x - n_x) ** 2 + (c_y - n_y) ** 2)

        if d < distance:
            distance = d
            final_coord = [c_x, c_y]
            pin_name = f"{col}{row+1}"

    print(final_coord)

    return int(final_coord[0]), int(final_coord[1]), pin_name

def translate(point, scale_to, expand_to, area_start):
    ''' 
        검출 좌표는 300, 300에서 찾아지므로 
        expand_to로 스케일 변환을 하여 조정할 필요가 있다.
        예측된 좌표를 homocoords로 변환하고
        스케일링 후 area_start 만큼 이동한다.

    '''
    point = xyToHomocoords(point[0], point[1])

    to_x_sf = expand_to / scale_to
    to_y_sf = expand_to / scale_to

    move = np.array([
        [1, 0, area_start[0]],
        [0, 1, area_start[1]],
        [0, 0, 1],
    ])

    scaling = np.array([
        [ to_x_sf,       0,   0],
        [       0, to_y_sf,   0],
        [       0,       0,   1]
    ])
    
    point = move @ scaling @ point 

    return homocoordsToxy(point.T)

def initializePinmaps(body_pinmap, vol_pinmap, transform_mtrx):
    global PADDING

    for C in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        for R in range(30):
            x, y = transform_pts(body_pinmap, transform_mtrx, C, R)
            body_pinmap.xs(R)[C]['x'] = round(x) + PADDING
            body_pinmap.xs(R)[C]['y'] = round(y) + PADDING

            # cv2.circle(result, (body_pinmap.xs(R)[C]['x'] - PADDING, body_pinmap.xs(R)[C]['y'] - PADDING), 10, (0, 255, 0), cv2.FILLED)

    for V in ['V1', 'V2', 'V3', 'V4']:
        for R in range(25):
            x, y = transform_pts(vol_pinmap, transform_mtrx, V, R)
            vol_pinmap.xs(R)[V]['x'] = round(x) + PADDING
            vol_pinmap.xs(R)[V]['y'] = round(y) + PADDING

            # cv2.circle(result, (vol_pinmap.xs(R)[V]['x'] - PADDING, vol_pinmap.xs(R)[V]['y'] - PADDING), 10, (0, 255, 255), cv2.FILLED)


def transform_pts(df, mtrx, col, row):
    fixed_x = df.xs(row)[col]['x']
    fixed_y = df.xs(row)[col]['y']

    n_point = [[fixed_x], [fixed_y], [1]]

    new_point = mtrx @ n_point

    new_x = new_point[0] / new_point[2]
    new_y = new_point[1] / new_point[2]

    return new_x[0], new_y[0]

def breadboard_bodypin_df(pinmap, padding=0):
    A = np.array(BREADBOARD_COL_INDEX)
    B = np.array(['x', 'y'] * 10)
    
    pinmap_pd = pd.DataFrame(columns = [A, B], index=BREADBOARD_ROW_INDEX)

    p2 = np.uint32(pinmap["2"]["points"]).reshape(5, 30, 2)
    p2[:, :, 0] += (pinmap["2"]["start"] + padding)
    p2[:, :, 1] += padding

    p3 = np.uint32(pinmap["3"]["points"]).reshape(5, 30, 2)
    p3[:, :, 0] += (pinmap["3"]["start"] + padding)
    p3[:, :, 1] += padding

    pinmap_pd['A'] = p2[0, :]
    pinmap_pd['B'] = p2[1, :]
    pinmap_pd['C'] = p2[2, :]
    pinmap_pd['D'] = p2[3, :]
    pinmap_pd['E'] = p2[4, :]

    pinmap_pd['F'] = p3[0, :]
    pinmap_pd['G'] = p3[1, :]
    pinmap_pd['H'] = p3[2, :]
    pinmap_pd['I'] = p3[3, :]
    pinmap_pd['J'] = p3[4, :]

    return pinmap_pd

def breadboard_voltagepin_df(pinmap, padding=0):
    A = np.array(['V1', 'V1', 'V2', 'V2', 'V3', 'V3', 'V4', 'V4'])
    B = np.array(['x', 'y'] * 4)
    
    v1 = np.uint32(pinmap["1"]["points"]).reshape(2, 25, 2)
    v1[:, :, 0] += (pinmap["1"]["start"] + padding)
    v1[:, :, 1] += padding

    v2 = np.uint32(pinmap["4"]["points"]).reshape(2, 25, 2)
    v2[:, :, 0] += (pinmap["4"]["start"] + padding)
    v2[:, :, 1] += padding

    pinmap_pd = pd.DataFrame(columns = [A, B], index=range(25))

    pinmap_pd['V1'] = v1[0, :]
    pinmap_pd['V2'] = v1[1, :]

    pinmap_pd['V3'] = v2[0, :]
    pinmap_pd['V4'] = v2[1, :]

    return pinmap_pd
