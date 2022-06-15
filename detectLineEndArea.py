import json
from re import A
from statistics import median
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN
from test_code.mappingDots import breadboard_bodypin_df, breadboard_voltagepin_df, transform_pts

MODEL_PATH = "model/breadboard-area.model.pt"
MODEL_LINEAREA_PATH = "model/line-area.model.pt"
MODEL_LINE_ENDPOINT_PATH = "model/line-endpoint.model.pt"

IMG = "images/Circuits/220504/Circuit_220504-32.jpeg" # -> OK
check_points = np.array([[ 404, 524], [ 412, 3692], [2512, 3664], [2488, 512]])

PADDING = 100

def findMaxArea(contours):
    max_contour = None
    max_area = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        _, _, w, h = cv2.boundingRect(cnt)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = cnt

    return max_area, max_contour

def toPerspectiveImage(img, points):
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
        [PADDING, PADDING],
        [width - 1 + PADDING, PADDING],
        [width - 1 + PADDING, height - 1 + PADDING],
        [PADDING, height - 1 + PADDING]
    ])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    return pts2, cv2.warpPerspective(img, mtrx, (width + 2*PADDING, height + 2*PADDING), flags=cv2.INTER_CUBIC)

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
    col_name = [index_map[idx] for idx in range_x]

    isVContains = ["V" in col for col in col_name]

    for vc in isVContains:
        if vc == False:
            pin_y = [search_map.iloc[row, range(5, 24, 2)].mean() - PADDING for row in range(30)]

        else:
            pin_y = [search_map.iloc[row, [1, 3, 25, 27]].mean() - PADDING for row in range(25)]
        
    pin_y = np.array(pin_y, np.float64)
    range_y = np.array(np.where(((pin_y >= area_start[1]) & (pin_y <= area_end[1]))))[0].tolist()
    row_name = [idx+1 for idx in range_y]

    for col in col_name:
        for row in row_name:
            table_idx.append(f"{col}{row}")

    return table_idx


if __name__ == "__main__":
    rng = 0.05

    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    base_point, target = toPerspectiveImage(target, check_points)
    base_point = np.uint32(base_point)
    # cv2.rectangle(target, (base_point[0]), (base_point[2]), (255, 0, 255), 15)

    pin_target = target.copy()

    pinmap = json.load(open("backend/static/data/pinmap.json"))

    pinmap_shape = pinmap["shape"]

    start_point = np.array([
        [0, 0], 
        [pinmap_shape[1], 0], 
        [pinmap_shape[1], pinmap_shape[0]],  
        [0, pinmap_shape[0]]
    ], dtype=np.float32)

    base_point = base_point.astype(np.float32)
    transform_mtrx = cv2.getPerspectiveTransform(start_point, base_point)

    body_pinmap = breadboard_bodypin_df(pinmap, PADDING)
    vol_pinmap = breadboard_voltagepin_df(pinmap, PADDING)

    base_point = np.uint32(base_point)
    pinmap_shape = pinmap["shape"]

    src_shape = (base_point[2][1] - base_point[0][1], base_point[2][0] - base_point[0][0])

    for C in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        for R in range(30):
            x, y = transform_pts(body_pinmap, transform_mtrx, C, R)
            body_pinmap.xs(R)[C]['x'] = round(x)
            body_pinmap.xs(R)[C]['y'] = round(y)

            # cv2.circle(target, (body_pinmap.xs(R)[C]['x'], body_pinmap.xs(R)[C]['y']), 15, (0, 20, 255), cv2.FILLED)

    for V in ['V1', 'V2', 'V3', 'V4']:
        for R in range(25):
            x, y = transform_pts(vol_pinmap, transform_mtrx, V, R)
            vol_pinmap.xs(R)[V]['x'] = round(x)
            vol_pinmap.xs(R)[V]['y'] = round(y)

            # cv2.circle(target, (vol_pinmap.xs(R)[V]['x'], vol_pinmap.xs(R)[V]['y']), 15, (25, 150, 255), cv2.FILLED)

    line_endpoint_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINE_ENDPOINT_PATH)

    # line-endpoint area 
    r = pd.DataFrame(line_endpoint_detect_model(pin_target).pandas().xyxy[0])
    
    palate = np.zeros((500, 500))
    palate_3d = np.zeros((500, 500, 3))

    breadboard_image_center = (pin_target.shape[1]/2, pin_target.shape[0]/2)

    for i in range(len(r)):
        data = r.iloc[i]
        if data.confidence > 0.5:
            p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)] # 검출된 전선 꼭지 영역 좌표

            start_, end_, pad_area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], expand_to = 300, blank=False)
            area = pad_area.copy()
            color_area = area.copy()

            cv2.rectangle(target, start_, end_, (0, 255, 0), 5)
            cv2.imshow(f"pad_area{i}", pad_area)
            '''
                영역안에 들어있는 후보핀 체크
            '''
            table_idx = findCandidateCoords(start_, end_, body_pinmap, vol_pinmap)
            '''
                영역안에 들어있는 후보핀 체크 코드 끝
            '''

            for pin in table_idx:
                if "V" in pin:
                    row = int(pin[2]) - 1
                    col = str(pin[:2])

                else:
                    row = int(pin[1:]) - 1
                    col = pin[0]

                print(row, col)

                getXY = (lambda df: (
                    df.xs(row)[col]['x'] - start_[0], 
                    df.xs(row)[col]['y'] - start_[1]))

                if "V" in col:
                    x, y = getXY(vol_pinmap)
                else:
                    x, y = getXY(body_pinmap)

                cv2.circle(area, (x - PADDING, y - PADDING), 5, (255, 0, 0), 10)
                # print(pin)
            
            cv2.imshow(f"LineEnd{i}", area)
            cv2.imwrite(f"LineEnd{i}.jpg", area)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            
            _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            area = cv2.morphologyEx(area, cv2.MORPH_CLOSE, kernel, iterations=7)

            ''' 클러스터링 시작 '''
            dbsc = DBSCAN(eps=1, min_samples=5, metric = 'euclidean', algorithm ='auto')
            
            areas = np.array(np.where(area != 0))
            dbsc.fit_predict(areas.T)

            labels = set(dbsc.labels_)

            max_area_label = -2
            max_area_width = -1
            for label in labels:
                segmentated = np.where(dbsc.labels_ == label)
                if (a:=len(segmentated[0])) > max_area_width:
                    max_area_label = label
                    max_area_width = a

            segmentated = np.array(list(set(np.where(dbsc.labels_ == max_area_label)[0]).intersection(dbsc.core_sample_indices_)))
            
            mask_img = np.zeros(area.shape)
            coords = areas.T[segmentated]

            for coord in coords:
                mask_img[coord[0], coord[1]] = 230

            cv2.imshow(f"threshold_origin_{i}", area)
            cv2.imshow(f"segmentation_{i}", mask_img)
            cv2.imwrite(f"threshold_origin_{i}.jpg", area)
            cv2.imwrite(f"segmentation_{i}.jpg", mask_img)
            ''' 클러스터링 끝 '''
          
    cv2.imshow("res", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
