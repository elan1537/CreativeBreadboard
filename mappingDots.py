import random
import numpy as np
import cv2
import json
import pandas as pd

# IMG = "./images/Circuits/220404/2_LB.jpeg"
# check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])

IMG = "./static/uploads/20220414_115935.jpg"
check_points = np.array([[ 676,  220], [ 668, 2724], [2320, 2736], [2332,  224]])

# IMG = "./static/uploads/IMG_4413.jpg"
# check_points = np.array([[ 500,  568], [ 488, 3692], [2520, 3696], [2580, 588]])

PADDING = 100

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

def circle_to_col(col, row):
    global base_point, pinmap_shape

    scale_x = (base_point[2][0] - base_point[0][0])/pinmap_shape[1]
    scale_y = (base_point[2][1] - base_point[0][1])/pinmap_shape[0]

    fixed_x = pinmap_pd.xs(row)[col]['x']
    fixed_y = pinmap_pd.xs(row)[col]['y']

    x = int((fixed_x + PADDING) * scale_x)
    y = int((fixed_y + PADDING) * scale_y)

    cv2.circle(target, (x, y), 15, (0, 20, 255), cv2.FILLED)


if __name__ == "__main__":
    global base_point, pinmap_shape
    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    
    # resize 후 핀맵에 매핑
    base_point, target = toPerspectiveImage(target, check_points)
    base_point = np.uint32(base_point)
    cv2.rectangle(target, (base_point[0]), (base_point[2]), (255, 0, 255), 15)

    print(base_point[0], base_point[2])
    print(target.shape)

    # 핀맵 캔버스 생성
    pinmap = json.load(open("./static/data/pinmap.json", "r"))
    pinmap_shape = pinmap["shape"]


    # 판다스로 저장하기
    A = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'F', 'F', 'G', 'G', 'H', 'H', 'I', 'I', 'J', 'J'])
    B = np.array(['x', 'y'] * 10)
    
    pinmap_pd = pd.DataFrame(columns = [A, B], index=range(30))

    p2 = np.uint32(pinmap["2"]["points"]).reshape(5, 30, 2)
    p2[:, :, 0] += pinmap["2"]["start"]

    p3 = np.uint32(pinmap["3"]["points"]).reshape(5, 30, 2)
    p3[:, :, 0] += pinmap["3"]["start"]


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

    for C in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        for R in range(30):
            circle_to_col(C, R)

    cv2.imshow('draw_pin in target', target)

    cv2.waitKey(0)
    cv2.destroyAllWindows()