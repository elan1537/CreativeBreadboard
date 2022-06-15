from matplotlib.pyplot import getp
from tensorflow.keras.models import load_model
import torch
import cv2
import json
import numpy as np
import pandas as pd
import math
from mappingDots import breadboard_bodypin_df, breadboard_voltagepin_df, transform_pts

MODEL_RESISTORAREA_PATH = "model/resistor-area.model.pt"

IMG = "images/Circuits/220428/Circuit-5.220428.jpg"
JSON = "images/Circuits/220428/Circuit-5.220428.json"
check_points = np.array([[ 596, 568], [ 620, 3288], [2312, 3176], [2440, 596]])

# IMG = "images/Circuits/220504/Circuit_220504-32.jpeg" # -> OK
# JSON = "images/Circuits/220504/Circuit_220504-32.json" # -> OK
# check_points = np.array([[ 404, 524], [ 412, 3692], [2512, 3664], [2488, 512]])

# IMG = "images/Circuits/220404/2_LB.jpeg" # -> OK
# JSON = "images/Circuits/220404/2_LB.json"
# check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])


PADDING = 200

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
    return pts1, pts2, mtrx, cv2.warpPerspective(img, mtrx, (width + 2*padding, height + 2*padding), flags=cv2.INTER_CUBIC)

def findCandidateCoords(area_start, area_end, bodymap, volmap):
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

def processDataFrame(origin_data, column_name):
    df = origin_data[(origin_data["name"] == column_name) & (origin_data["confidence"] > 0.7)].copy()
    df['area']      = df.apply(rectArea, axis=1)
    df['center_x']  = df.apply(lambda row: int((row.xmax + row.xmin) / 2), axis=1)
    df['center_y']  = df.apply(lambda row: int((row.ymax + row.ymin) / 2), axis=1)
    df['length']    = df.apply(lambda row: int((row.xmax - row.xmin)), axis=1)
    df['width']     = df.apply(lambda row: int((row.ymax - row.ymin)), axis=1)
    df['distance_from_origin'] = df.apply(lambda row: int((row.xmin + row.ymin)), axis=1)
    df = df.sort_values(by=['distance_from_origin'], ascending=True)

    return df

def rectArea(df):
    return (df.xmax - df.xmin) * (df.ymax - df.ymin)

def xyToHomocoords(x, y):
    return np.array([
        [x],
        [y],
        [1]
    ])
            
def homocoordsToxy(v):
    return int(v[0][0]/v[0][2]), int(v[0][1]/v[0][2])

def imgNormalizing(src):
    img_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img = cv2.resize(img_output, (300, 300))
    img = img.reshape(-1, 300, 300, 3)

    mean = np.mean(img, axis=(0, 1, 2, 3))
    std = np.std(img, axis=(0, 1, 2, 3))

    img = (img-mean)/(std + 1e-7)
    return img

def getXYPinCoords(model, src):
    if model:
        c = list(model.predict(src)[0])
        return c

def getPinCoords(search_map, candidates, coord, area_start):
    distance = 1000000
    final_coord = []
    pin_name = None

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

        try:
            cv2.putText(area, f"{col}_{row}", (round(c_x), round(c_y)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(area, (round(c_x), round(c_y)), 10, (10, 50, 80), cv2.FILLED)
            cv2.circle(area, (round(n_x), round(n_y)), 10, (255, 255, 80), cv2.FILLED)
        except:
            pass

        d = math.sqrt((c_x - n_x) ** 2 + (c_y - n_y) ** 2)

        if d < distance:
            distance = d
            final_coord = [c_x, c_y]
            pin_name = f"{col}{row+1}"

    return int(final_coord[0]), int(final_coord[1]), pin_name

def translate(point, expand_to):
    ''' 
        검출 좌표는 300, 300에서 찾아지므로 
        expand_to로 스케일 변환을 하여 조정할 필요가 있다.
        예측된 좌표를 homocoords로 변환하고
        스케일링 후 area_start 만큼 이동한다.

    '''
    point = xyToHomocoords(point[0], point[1])
    g = xyToHomocoords(pt2[0], pt2[1])

    to_x_sf = expand_to / 300
    to_y_sf = expand_to / 300

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
    
    p = move @ scaling @ p 
    g = move @ scaling @ g

    return homocoordsToxy(p.T)

def initializePinmaps(body_pinmap, vol_pinmap):
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
    

if __name__ == "__main__":
    model = load_model("/Users/se_park/Library/Mobile Documents/com~apple~CloudDocs/2022 Soongsil/1. CS/CreativeBreadboard/model/findCoordsResNet50.h5")
    
    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    _, base_point, mtrx, result = toPerspectiveImage(target, check_points, PADDING)
    pin_target = result.copy()

    pinmap = json.load(open("backend/static/data/pinmap.json"))

    # first generate pinmaps
    body_pinmap = breadboard_bodypin_df(pinmap, PADDING)
    vol_pinmap = breadboard_voltagepin_df(pinmap, PADDING)

    # fit to target image
    base_point = np.uint32(base_point)
    pinmap_shape = pinmap["shape"]

    start_point = np.array([
        [PADDING, PADDING], 
        [pinmap_shape[1] + PADDING, PADDING], 
        [pinmap_shape[1] + PADDING, pinmap_shape[0] + PADDING],  
        [PADDING, pinmap_shape[0] + PADDING]
    ], dtype=np.float32)

    base_point = base_point.astype(np.float32)
    transform_mtrx = cv2.getPerspectiveTransform(start_point, base_point)

    initializePinmaps(body_pinmap, vol_pinmap)

    search_map = pd.concat([vol_pinmap.iloc[:, 0:4], body_pinmap, vol_pinmap.iloc[:, 4:8]], axis=1)
    
    resistor_detect_area = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_RESISTORAREA_PATH)
    detect_area = pd.DataFrame(resistor_detect_area(result).pandas().xyxy[0])
    resistor_area = processDataFrame(detect_area, "resistor-area")


    for i in range(len(resistor_area)):
        data = resistor_area.iloc[i]

        minPoint = round(data.xmin)-15, round(data.ymin)-15
        maxPoint = round(data.xmax)+15, round(data.ymax)+15

        expand_to = max([maxPoint[0] - minPoint[0], maxPoint[1] - minPoint[1]])
        area_start, area_end, area = area_padding(result, minPoint, maxPoint, base_point[0], base_point[2], expand_to, True)

        table_idx = findCandidateCoords(area_start, area_end, body_pinmap, vol_pinmap)

        normalized = imgNormalizing(area)
        coords = getXYPinCoords(model, normalized)

        pt1 = round(coords[0]), round(coords[1])
        pt2 = round(coords[2]), round(coords[3])

        pt1 = translate(pt1, expand_to)
        pt2 = translate(pt2, expand_to)

        x1, y1, pin1 = getPinCoords(search_map, table_idx, pt1, area_start)
        x2, y2, pin2 = getPinCoords(search_map, table_idx, pt2, area_start)

        cv2.putText(result, pin1, (x1 + area_start[0], y1 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.circle(result, (x1 + area_start[0], y1 + area_start[1]), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(result, pin2, (x2 + area_start[0], y2 + area_start[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.circle(result, (x2 + area_start[0], y2 + area_start[1]), 10, (20, 0, 255), cv2.FILLED)

        cv2.imshow(f"area{i}", area)
        cv2.imwrite(f"area{i}.jpg", area)

    del model
    del resistor_detect_area

    cv2.imshow(f"result", result)
    cv2.imwrite("result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 