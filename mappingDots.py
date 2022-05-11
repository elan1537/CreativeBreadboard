import random
import numpy as np
import cv2
import json
import pandas as pd

IMG = "./images/Circuits/220404/2_LB.jpeg"
check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])

# IMG = "./static/uploads/20220414_115935.jpg"
# check_points = np.array([[ 676,  220], [ 668, 2724], [2320, 2736], [2332,  224]])

# IMG = "./static/uploads/IMG_4413.jpg"
# check_points = np.array([[ 500,  568], [ 488, 3692], [2520, 3696], [2580, 588]])

PADDING = 100

BREADBOARD_COL_INDEX = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'F', 'F', 'G', 'G', 'H', 'H', 'I', 'I', 'J', 'J']
BREADBOARD_ROW_INDEX = range(30)

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

def transform_pts(df, src_shape:tuple , dst_shape:tuple , padding, col, row):
    scale_x = src_shape[1] / dst_shape[1]
    scale_y = src_shape[0] / dst_shape[0]

    fixed_x = df.xs(row)[col]['x']
    fixed_y = df.xs(row)[col]['y']

    x = int((fixed_x + padding) * scale_x)
    y = int((fixed_y + padding) * scale_y)

    return x, y


def circle_to_col(df: pd.DataFrame, col, row):
    '''df: DataFrame으로 생성된 핀 좌표 테이블'''

    global base_point, pinmap_shape
    src_shape = (base_point[2][1] - base_point[0][1], base_point[2][0] - base_point[0][0])

    x, y = transform_pts(df, src_shape, pinmap_shape, PADDING, col, row)
    cv2.circle(target, (x, y), 15, (0, 20, 255), cv2.FILLED)


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

if __name__ == "__main__":
    global base_point, pinmap_shape
    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    
    # resize 후 핀맵에 매핑
    base_point, target = toPerspectiveImage(target, check_points)
    base_point = np.uint32(base_point)
    cv2.rectangle(target, (base_point[0]), (base_point[2]), (255, 0, 255), 15)

    # 핀맵 캔버스 생성
    pinmap = json.load(open("./static/data/pinmap.json", "r"))
    pinmap_shape = pinmap["shape"]

    body_pinmap = breadboard_bodypin_df(pinmap)
    vol_pinmap = breadboard_voltagepin_df(pinmap)

    print(body_pinmap)

    for C in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        for R in range(30):
            circle_to_col(body_pinmap, C, R)

    for V in ['V1', 'V2', 'V3', 'V4']:
        for R in range(25):
            circle_to_col(vol_pinmap, V, R)

    cv2.imshow('draw_pin in target', target)

    cv2.waitKey(0)
    cv2.destroyAllWindows()