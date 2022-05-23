import pandas as pd
import json
import cv2
import numpy as np
import torch
import os
import shutil
import time

PADDING = 100

# IMG = "test_code/Circuit-5.220428.jpg"
# JSON = "test_code/Circuit-5.220428.json"
# check_points = np.array([[ 596, 568], [ 620, 3288], [2312, 3176], [2440, 596]])

# IMG = "images/Circuits/220404/2_LB.jpeg" # -> OK
# JSON = "images/Circuits/220404/2_LB.json"
# check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])

# IMG = "images/Circuits/220504/Circuit_220504-32.jpeg" # -> OK
# JSON = "images/Circuits/220504/Circuit_220504-32.json" # -> OK
# check_points = np.array([[ 404, 524], [ 412, 3692], [2512, 3664], [2488, 512]])

'''
46
'''

ORIGIN_IMG_PATH = "backend/static/uploads/origin_img"
CHECK_POINT_PATH = "backend/static/uploads/check_points"
ANNOTATION_PATH = "backend/static/uploads/annotation"

IMGS = [ ORIGIN_IMG_PATH + "/" + img for img in os.listdir(ORIGIN_IMG_PATH) if ".jpeg" in img or ".JPG" in img or ".jpg" in img]
CHECK_POINTS = [ CHECK_POINT_PATH + "/" +  point for point in os.listdir(CHECK_POINT_PATH) if ".json" in point ]
ANNOTATION_DATA = [ ANNOTATION_PATH + "/" + anno for anno in os.listdir(ANNOTATION_PATH) if ".json" in anno ]

IMGS = sorted(IMGS)
CHECK_POINTS = sorted(CHECK_POINTS)
ANNOTATION_DATA = sorted(ANNOTATION_DATA)

MODEL_RESISTORAREA_PATH = "model/resistor-area.model.pt"
DATASET = "test_code/dataset/resistor_point"
SCALE_TO = 300

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

def main():
    shutil.rmtree("/Users/se_park/Library/Mobile Documents/com~apple~CloudDocs/2022 Soongsil/1. CS/CreativeBreadboard/test_code/dataset/resistor_point")

    RES_AREA_COUNT = 0
    files = None
    if not os.path.isdir(f"{DATASET}"):
        os.mkdir(f"{DATASET}") 

    else:
        files = [f for f in os.listdir(DATASET) if '.jpg' in f or ".jpeg" in f or ".JPG" in f]

    try:
        training_data = json.load(open(f"{DATASET}/points.json"))
    except FileNotFoundError as fn:
        training_data = {}

    start_idx = 1

    print("COUNTS:: ", len(IMGS))

    resistor_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_RESISTORAREA_PATH)

    for IMG, ANNO_JSON, ROI_JSON in zip(IMGS, ANNOTATION_DATA, CHECK_POINTS):
        img = cv2.imread(IMG, cv2.IMREAD_COLOR)
        shapes = json.load(open(ANNO_JSON, "r"))["shapes"]
        check_points = np.array(json.load(open(ROI_JSON, "r")))
        shapes = pd.DataFrame(shapes)
        
        resistor_vector = shapes[shapes['label'] == 'resistor-vector']

        # resistor_vector가 없어..?
        if len(resistor_vector) == 0:
            print(f"{ROI_JSON.split('/')[-1]} is no resistor_vector")
            continue

        _, base_point, mtrx, result = toPerspectiveImage(img, check_points, PADDING)
        detect_area = pd.DataFrame(resistor_detect_model(result).pandas().xyxy[0])  

        # resistor_body = processDataFrame(detect_area, "resistor-body")
        resistor_area = processDataFrame(detect_area, "resistor-area")
        print("resistor_area: ", len(resistor_area), end='')
        print(", summation res_count::", RES_AREA_COUNT)
        
        RES_AREA_COUNT += len(resistor_area)

        for i in range(len(resistor_area)):
            data = resistor_area.iloc[i]
            minPoint = round(data.xmin)-15, round(data.ymin)-15
            maxPoint = round(data.xmax)+15, round(data.ymax)+15

            expand_to = max([maxPoint[0] - minPoint[0], maxPoint[1] - minPoint[1]])

            # cv2.rectangle(result, minPoint, maxPoint, (255, 0, 0), 5)
            area_start, area_end, area = area_padding(result, minPoint, maxPoint, base_point[0], base_point[2], expand_to, True)
            area_copy = area.copy()

            for j in range(len(resistor_vector)):
                pts = resistor_vector['points'].iloc[j]

                for pt in pts:
                    x = round(pt[0])
                    y = round(pt[1])

                    p = xyToHomocoords(x, y)
                    p = (mtrx @ p)
                    x, y = homocoordsToxy(p.T)

                    if (area_start[0] <= x and x < area_end[0]) and (area_start[1] <= y and y < area_end[1]):
                        # cv2.circle(result, (x, y), 15, (255, 0, 255), cv2.FILLED)
                        offset = np.array([
                            [1, 0, -area_start[0]],
                            [0, 1, -area_start[1]],
                            [0, 0, 1]
                        ])
                        p = offset @ p

                        area_copy = cv2.resize(area_copy, (SCALE_TO, SCALE_TO))
                        to_x_sf = SCALE_TO / expand_to
                        to_y_sf = SCALE_TO / expand_to

                        scaling = np.array([
                            [ to_x_sf,       0,  (1-to_x_sf)],
                            [       0, to_y_sf,  (1-to_y_sf)],
                            [       0,       0,            1]
                        ])

                        p = scaling @ p

                        x, y = homocoordsToxy(p.T)     
                        
                        # 검은색 영역엔 점을 찍지 않는다.
                        # if area[y, x, 0] == 0 and area[y, x, 1] == 0 and area[y, x, 2] == 0:
                        #     print("         In Blank area")
                        # else:
                        # cv2.circle(area_copy, (x, y), 7, (255, 0, 0), 10)
                        
                        if start_idx > 0 and start_idx < 10:
                            if training_data.get(f"resistor-0{start_idx}") != None:
                                training_data[f"resistor-0{start_idx}"].append([x, y])
                            else:
                                training_data[f"resistor-0{start_idx}"] = [[x, y]]
                            print(f"         saved_at :: resistor_0{start_idx}")

                        else:
                            if training_data.get(f"resistor-{start_idx}") != None:
                                training_data[f"resistor-{start_idx}"].append([x, y])
                            else:
                                training_data[f"resistor-{start_idx}"] = [[x, y]]
                            print(f"         saved_at :: resistor_{start_idx}")


            if start_idx > 0 and start_idx < 10:      
                cv2.imwrite(f"{DATASET}/resistor-0{start_idx}.jpg", area_copy)
            else:
                cv2.imwrite(f"{DATASET}/resistor-{start_idx}.jpg", area_copy)

            # cv2.imshow(f"area_copy{start_idx}", area_copy)
            start_idx += 1

        # cv2.imshow(f"{IMG.split('/')[-1]}", result)
    with open(f"{DATASET}/points.json", "w") as f:
        json.dump(training_data, f)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()