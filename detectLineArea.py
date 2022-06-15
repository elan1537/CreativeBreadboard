import random
from statistics import median
import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

MODEL_PATH = "./model/breadboard-area.model.pt"
MODEL_LINEAREA_PATH = "./model/line-area.best.pt"
MODEL_LINE_ENDPOINT_PATH = "./model/line-endpoint.best.pt"
# IMG = "./images/Circuits/220428/Circuit-12.220428.jpg"
# IMG = "./images/Circuits/220428/Circuit-7.220428.jpg"
# IMG = "./images/res.jpg" # 브레드보드만 딴 이미지

IMG = "./images/Circuits/220404/2_LB.jpeg"
JSON = "./images/Circuits/220404/2_LB.json"
check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])

# IMG = "images/Circuits/220504/Circuit_220504-32.jpeg" # -> OK
# check_points = np.array([[ 404, 524], [ 412, 3692], [2512, 3664], [2488, 512]])

IMG = "images/Circuits/220414/20220414_115935.jpg"
check_points = np.array([[ 676,  220], [ 668, 2724], [2320, 2736], [2332,  224]])

# IMG = "images/Circuits/220428/Circuit-5.220428.jpg"
# # JSON = "test_code/Circuit-5.220428.json"
# check_points = np.array([[ 596, 568], [ 620, 3288], [2312, 3176], [2440, 596]])


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

def color_test(windowName:str, src):
    area_gray_color = src.copy()
    area_gray_color = cv2.cvtColor(area_gray_color, cv2.COLOR_BGR2GRAY)
    
    hsv_area = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_area)
    equalizedV = cv2.equalizeHist(v)
    hsv_area = cv2.merge([h,s,equalizedV])
    hsv_to_bgr = cv2.cvtColor(hsv_area, cv2.COLOR_HSV2BGR)
    
    cv2.imshow(f"before eq and after eq_{i}", np.hstack([src, hsv_to_bgr]))

    nor = cv2.equalizeHist(area_gray_color)

    # hist = cv2.calcHist([area_keypoint],[0],None,[256],[0,256])
    # cv2.normalize(hist, hist, 0, area_keypoint.shape[0], cv2.NORM_MINMAX)

    # print(hist)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    gr = cv2.morphologyEx(area_gray_color, cv2.MORPH_GRADIENT, kernel, iterations=3)
    op = cv2.morphologyEx(area_gray_color, cv2.MORPH_OPEN, kernel, iterations=3)
    cl = cv2.morphologyEx(area_gray_color, cv2.MORPH_CLOSE, kernel, iterations=3)
    er = cv2.morphologyEx(area_gray_color, cv2.MORPH_ERODE, kernel, iterations=3)

    adapt_gr = cv2.adaptiveThreshold(gr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 15)
    adapt_op = cv2.adaptiveThreshold(op, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 15)
    adapt_cl = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 15)
    adapt_er = cv2.adaptiveThreshold(er, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 15)

    _, th_gr = cv2.threshold(gr, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    _, th_op = cv2.threshold(op, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    _, th_cl = cv2.threshold(cl, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    _, th_er = cv2.threshold(er, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    cv2.imshow(windowName, np.vstack(
        [   
            np.hstack([area_gray_color, nor,          gr,       op,       cl,       er]),
            np.hstack([area_gray_color, nor,       th_gr,    th_op,    th_cl,    th_er]),
            np.hstack([area_gray_color, nor,    adapt_gr, adapt_op, adapt_cl, adapt_er])
        ]))

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
    rng = 0.05
    PADDING = 500

    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    # shapes = json.load(open(JSON, "r"))["shapes"]
    # shapes = pd.DataFrame(shapes)

    line_area_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEAREA_PATH)
    line_endpoint_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINE_ENDPOINT_PATH)

    _, base_point, mtrx, target = toPerspectiveImage(target, check_points, PADDING)

    detect_line_area = pd.DataFrame(line_area_detect_model(target).pandas().xyxy[0])
    detect_line_endarea = pd.DataFrame(line_endpoint_detect_model(target).pandas().xyxy[0])

    line_area = processDataFrame(detect_line_area, "line-area")
    line_endarea = processDataFrame(detect_line_endarea, "line-endpoint")

    key_table = line_contains_table(line_area, line_endarea)

    key_table = pd.Series(key_table)

    for la_num in set(key_table.values):
        line_e = key_table[key_table == la_num]

        R = int(random.random() * 255)
        G = int(random.random() * 255)
        B = int(random.random() * 255)

        line_a = line_area.iloc[la_num]

        minPointLine = round(line_a.xmin), round(line_a.ymin)
        maxPointLine = round(line_a.xmax), round(line_a.ymax)

        cv2.rectangle(target, minPointLine , maxPointLine , (B, G, R), 10, cv2.FILLED)

        for areaIdx, line_a in line_e.iteritems():
            line_e = line_endarea.iloc[areaIdx]

            minPointEnd = round(line_e.xmin), round(line_e.ymin)
            maxPointEnd = round(line_e.xmax), round(line_e.ymax)

            cv2.rectangle(target, minPointEnd , maxPointEnd , (B, G, R), 10, cv2.FILLED)


    cv2.imshow("Res", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()