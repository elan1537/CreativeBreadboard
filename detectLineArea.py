import random
from statistics import median
import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
import json

MODEL_PATH = "./model/breadboard-area.model.pt"
MODEL_LINEAREA_PATH = "./model/line-area.model.pt"
MODEL_LINE_ENDPOINT_PATH = "./model/line-endpoint.model.pt"
# IMG = "./images/Circuits/220428/Circuit-12.220428.jpg"
# IMG = "./images/Circuits/220428/Circuit-7.220428.jpg"
# IMG = "./images/res.jpg" # 브레드보드만 딴 이미지

IMG = "./images/Circuits/220404/2_LB.jpeg"
check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])

# IMG = "./static/uploads/20220414_115935.jpg"
# check_points = np.array([[ 676,  220], [ 668, 2724], [2320, 2736], [2332,  224]])

PADDING = 500

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

if __name__ == "__main__":
    rng = 0.05

    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    _, target = toPerspectiveImage(target, check_points)
    pin_target = target.copy()

    line_area_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINEAREA_PATH)
    line_endpoint_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINE_ENDPOINT_PATH)
    detect_area = pd.DataFrame(line_area_detect_model(target).pandas().xyxy[0])

    # line-endpoint area 
    r = pd.DataFrame(line_endpoint_detect_model(target).pandas().xyxy[0])

    for i in range(len(detect_area)):
        data = detect_area.iloc[i]

        if data.confidence < 0.8:
            continue

        R: int = random.randint(0, 255)
        G: int = random.randint(0, 255)
        B: int = random.randint(0, 255)

        p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]
        cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (B, G, R), 15)

        d_x = int(((p[2] - p[0]) / 2) * rng)
        d_y = int((p[3] - p[1]) / 2 * rng)

        p[0] -= d_x
        p[1] -= d_y
        p[2] += d_x
        p[3] += d_y

        if p[0] < 0:
            p[0] = 0

        if p[1] < 0:
            p[1] = 0

        if p[2] > int(data.xmax):
            p[2] = int(data.xmax)
        
        if p[3] > int(data.ymax):
            p[3] = int(data.ymax)

        area = target[p[1]:p[3], p[0]:p[2]]
        cv2.imshow(f"a_{i}", area)

    cv2.imshow("Res", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()