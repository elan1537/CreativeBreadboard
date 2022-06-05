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
# IMG = "./images/Circuits/220428/Circuit-12.220428.jpg"
# IMG = "./images/Circuits/220428/Circuit-7.220428.jpg"
# IMG = "./images/res.jpg" # 브레드보드만 딴 이미지

# IMG = "./static/uploads/IMG_4413.jpg" # -> OK
# check_points = np.array([[ 500,  568], [ 488, 3692], [2520, 3696], [2580, 588]])

# IMG = "images/Circuits/220414/20220414_115935.jpg" # -> ERROR .. 타겟 영역이 오른쪽 핀 영역까지 침범함 -> 해결
# check_points = np.array([[ 676,  220], [ 668, 2724], [2320, 2736], [2332,  224]])

IMG = "images/Circuits/220404/2_LB.jpeg" # -> OK
check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])

# IMG = "./static/uploads/Circuit_220504-32.jpeg" # -> OK
# check_points = np.array([[ 404, 524], [ 412, 3692], [2512, 3664], [2488, 512]])

PADDING = 0

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

def area_padding(old_area, from_: tuple, to_: tuple, canvas_start: tuple or list, canvas_to: tuple or list, expand_to = 0):
    '''
        타겟 영역에서 직사각형 영역 from_에서 to_ 까지 crop한다.
        padding이 이뤄진 전체 영역인 canvas_start, canvas_to를 가지고
        새롭게 crop 하는영역이 관심영역 (브레드보드 영역 안쪽)에만 잘리게 한다.

        expand_to로 주위 핀을 찾기위해 확장한다.
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

    # 110, 154 -> 350, 350
    # (350 - 110)/2 = 120, (350-154)/2 = 98
    if expand_to != 0:
        add_width  = int((expand_to - to_area.shape[1]) / 2)
        add_height = int((expand_to - to_area.shape[0]) / 2)

        x_[0] -= add_width
        y_[0] -= add_height
        x_[1] += add_width
        y_[1] += add_height

        # padding으로 확대했는데 범위를 넘어가면..
        # 범위를 넘어간 만큼 가능한 공간에서 다시 재확장된다.
        if x_[1] > canvas_to[0]:
            x_[1] -= add_width
            x_[0] -= add_width

        if x_[0] < canvas_start[0]:
            x_[0] += add_width
            x_[1] += add_width

        if y_[1] > canvas_to[1]:
            y_[1] -= add_height
            y_[0] -= add_height
        
        if y_[0] < canvas_start[1]:
            y_[1] += add_height
            y_[0] += add_height 

    # padding 만큼 확장된 결과
    expanded = old_area[y_[0]:y_[1], x_[0]:x_[1]]

    return (x_[0], y_[0]), (x_[1], y_[1]), expanded

if __name__ == "__main__":
    rng = 0.05

    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    base_point, target = toPerspectiveImage(target, check_points)
    base_point = np.uint32(base_point)
    # cv2.rectangle(target, (base_point[0]), (base_point[2]), (255, 0, 255), 15)

    pin_target = target.copy()

    pinmap = json.load(open("backend/static/data/pinmap.json"))

    body_pinmap = breadboard_bodypin_df(pinmap, PADDING)
    vol_pinmap = breadboard_voltagepin_df(pinmap, PADDING)

    base_point = np.uint32(base_point)
    pinmap_shape = pinmap["shape"]

    src_shape = (base_point[2][1] - base_point[0][1], base_point[2][0] - base_point[0][0])

    # cv2.imshow('no_map', pin_target)

    for C in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        for R in range(30):
            x, y = transform_pts(body_pinmap, src_shape, pinmap_shape, 0, C, R)
            body_pinmap.xs(R)[C]['x'] = x
            body_pinmap.xs(R)[C]['y'] = y

            # cv2.circle(target, (body_pinmap.xs(R)[C]['x'], body_pinmap.xs(R)[C]['y']), 15, (0, 20, 255), cv2.FILLED)

    for V in ['V1', 'V2', 'V3', 'V4']:
        for R in range(25):
            x, y = transform_pts(vol_pinmap, src_shape, pinmap_shape, 0, V, R)
            vol_pinmap.xs(R)[V]['x'] = x
            vol_pinmap.xs(R)[V]['y'] = y

            # cv2.circle(target, (x, y), 15, (25, 150, 255), cv2.FILLED)

    # cv2.imshow("pin_target", target)

    line_endpoint_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINE_ENDPOINT_PATH)

    # line-endpoint area 
    r = pd.DataFrame(line_endpoint_detect_model(pin_target).pandas().xyxy[0])
    
    palate = np.zeros((500, 500))
    palate_3d = np.zeros((500, 500, 3))

    breadboard_image_center = pin_target.shape[1], pin_target.shape[0]

    for i in range(len(r)):
        data = r.iloc[i]
        if data.confidence > 0.5:
            p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)] # 검출된 전선 꼭지 영역 좌표

            start_, end_, pad_area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], expand_to = 320)
            area = pad_area.copy()
            # cv2.imshow(f"areawraew_{i}", area)
            color_area = area.copy()

            cv2.rectangle(target, start_, end_, (0, 255, 0), 5)


            '''
                영역안에 들어있는 후보핀 체크
            '''
            search_map = None
            cols = None
            x_s = None
            idx = None

            if p[2] > breadboard_image_center[0]/2:
                search_map = body_pinmap.loc[:, "F":"J"]
                cols = ["F", "G", "H", "I", "J", "PIN_END", "V3", "V4", "END"]  

                x_s  = [ int(breadboard_image_center[0]/2) ]
                x_s += [ int(search_map[col, "x"].median()) for col in cols[:5] ]
                x_s += [ int((x_s[-1] + median(vol_pinmap.loc[:, "V3"]["x"].tolist()))/2) ]
                x_s += [ median(vol_pinmap.loc[:, "V3"]["x"].tolist()), median(vol_pinmap.loc[:, "V4"]["x"].tolist()), breadboard_image_center[0] ]

                # 그럼 J 부터 X까지도 있겠지..
                x_s = np.array(x_s, np.uint32)
                cols = np.array(cols, np.str0)
                idx = np.array(np.where((x_s > start_[0]) & (x_s < end_[0]))[0]) - 1
                # idx = np.append(idx, [idx[-1]+1])

            else:
                search_map = body_pinmap.loc[:, "A":"E"]
                cols = [ "START", "V1", "V2", "PIN_START", "A", "B", "C", "D", "E", "PIN_END"]

                x_s  = [ 0, median(vol_pinmap.loc[:, "V1"]["x"].tolist()), median(vol_pinmap.loc[:, "V2"]["x"].tolist()) ]
                x_s += [ int((median(vol_pinmap.loc[:, "V2"]["x"].tolist()) + int(search_map[cols[4], "x"].median())) / 2) ]
                x_s += [ int(search_map[col, "x"].median()) for col in cols[4:9] ]
                x_s += [ breadboard_image_center[0]/2 - 1 ]

                # 0 부터 A 전까지 범위는 어디에 속할까? 
                # 결국 전압핀맵...

                x_s = np.array(x_s, np.uint32)
                idx = np.array(np.where((x_s > start_[0]) & (x_s < end_[0]))[0])
                cols = np.array(cols, np.str0)

            x_res = cols[idx]

            if 'V1' not in x_res and 'V2' not in x_res and 'V3' not in x_res and 'V4' not in x_res:
                if len(x_res):
                    cols = cols[idx].tolist()

                    rows = np.arange(30, dtype=np.uint32)
                    y_s = [PADDING]
                    y_s += search_map[cols[0], "y"].tolist()

                    y_s = np.array(y_s, np.uint32)
                    idx = np.where((y_s > start_[1]) & (y_s < end_[1]))[0]
                    cols = np.array(idx, np.str0).tolist()
                    
                    table_idx = [x + r for x in x_res if x != "PIN_END" and x != "PIN_START" for r in cols ] 
                    print(table_idx)

                    # # border 만들고 여기다 drawing
                    # for x_border in x_s_border:
                    #     x_border = x_border - start_[0]
                    #     cv2.line(pad_area, (x_border, 0), (x_border, 350), (255, 10, 10), 5)

                    # for y_border in y_s_border:
                    #     y_border = y_border - start_[1]
                    #     cv2.line(pad_area, (0, y_border), (350, y_border), (53, 100, 20), 5)
            else:
                if len(x_res):
                    cols = cols[idx].tolist()

                    search_map = vol_pinmap.loc[:, cols[0]]

                    rows = np.arange(25, dtype=np.uint32)
                    y_s = [ PADDING ]
                    y_s += search_map["y"].tolist()

                    y_s = np.array(y_s, np.uint32)
                    idx = np.where((y_s > start_[1]) & (y_s < end_[1]))[0]
                    cols = np.array(idx, np.str0).tolist()

                    table_idx = [x + r for x in x_res if x != "PIN_END" and x != "PIN_START" for r in cols ] 
                    print(table_idx)


            '''
                영역안에 들어있는 후보핀 체크 코드 끝
            '''

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            area = cv2.morphologyEx(area, cv2.MORPH_OPEN, kernel, iterations=7)

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
            ''' 클러스터링 끝 '''
          
    # cv2.imshow("res", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()