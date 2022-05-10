import json
import numpy as np
import cv2


IMG = "./static/uploads/IMG_4413.jpg"
check_points = np.array([[ 500,  568], [ 488, 3692], [2520, 3696], [2580, 588]])

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

if __name__ == "__main__":
    canvas = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    
    # resize 후 핀맵에 매핑
    base_point, target = toPerspectiveImage(canvas, check_points)
    base_point = np.uint32(base_point)
    cv2.rectangle(target, (base_point[0]), (base_point[2]), (255, 0, 255), 15)
    pinmap = json.load(open("./static/data/pinmap.json", "r"))

    for pin in pinmap:
        if pin == "shape":
            continue

        target = pinmap[pin]
        points = np.uint32(sorted(target["points"])).reshape(5, -1, 2)

        
        temp = []
        for group in points:
            group = sorted(group.tolist(), key = lambda x: [x[1]])
            temp.append(group)

        pinmap[pin]["points"] = np.uint32(temp).reshape(-1, 2).tolist()

    with open("pinmap.json", "w") as f:
        json.dump(pinmap, f)

