from audioop import findmax
import json
from statistics import median
import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from mappingDots import breadboard_bodypin_df, breadboard_voltagepin_df, transform_pts

MODEL_PATH = "./model/breadboard-area.model.pt"
MODEL_LINEAREA_PATH = "./model/line-area.model.pt"
MODEL_LINE_ENDPOINT_PATH = "./model/line-endpoint.model.pt"
# IMG = "./images/Circuits/220428/Circuit-12.220428.jpg"
# IMG = "./images/Circuits/220428/Circuit-7.220428.jpg"
# IMG = "./images/res.jpg" # 브레드보드만 딴 이미지

# IMG = "./static/uploads/IMG_4413.jpg"
# check_points = np.array([[ 500,  568], [ 488, 3692], [2520, 3696], [2580, 588]])

# IMG = "./static/uploads/20220414_115935.jpg"
# check_points = np.array([[ 676,  220], [ 668, 2724], [2320, 2736], [2332,  224]])

# IMG = "./images/Circuits/220404/2_LB.jpeg"
# check_points = np.array([[ 544,  704], [ 528, 3620], [2376, 3576], [2252,  876]])

IMG = "./static/uploads/Circuit_220504-32.jpeg"
check_points = np.array([[ 404, 524], [ 412, 3692], [2512, 3664], [2488, 512]])

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
    # 범위를 넘어가나?
    x_ = [from_[0], to_[0]]
    y_ = [from_[1], to_[1]]

    print(from_, to_, canvas_start, canvas_to)

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

    return expanded

if __name__ == "__main__":
    rng = 0.05

    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    base_point, target = toPerspectiveImage(target, check_points)
    base_point = np.uint32(base_point)
    # cv2.rectangle(target, (base_point[0]), (base_point[2]), (255, 0, 255), 15)

    pin_target = target.copy()

    pinmap = json.load(open("./static/data/pinmap.json"))

    body_pinmap = breadboard_bodypin_df(pinmap, PADDING)
    vol_pinmap = breadboard_voltagepin_df(pinmap, PADDING)

    base_point = np.uint32(base_point)
    pinmap_shape = pinmap["shape"]

    src_shape = (base_point[2][1] - base_point[0][1], base_point[2][0] - base_point[0][0])
    print(src_shape)

    for C in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        for R in range(30):
            x, y = transform_pts(body_pinmap, src_shape, pinmap_shape, 0, C, R)
            body_pinmap.xs(R)[C]['x'] = x
            body_pinmap.xs(R)[C]['y'] = y

            cv2.circle(target, (body_pinmap.xs(R)[C]['x'], body_pinmap.xs(R)[C]['y']), 15, (0, 20, 255), cv2.FILLED)

    for V in ['V1', 'V2', 'V3', 'V4']:
        for R in range(25):
            x, y = transform_pts(vol_pinmap, src_shape, pinmap_shape, 0, V, R)
            vol_pinmap.xs(R)[V]['x'] = x
            vol_pinmap.xs(R)[V]['y'] = y

            cv2.circle(target, (x, y), 15, (0, 20, 255), cv2.FILLED)

    line_endpoint_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LINE_ENDPOINT_PATH)

    # line-endpoint area 
    r = pd.DataFrame(line_endpoint_detect_model(cv2.cvtColor(pin_target, cv2.COLOR_BGR2GRAY)).pandas().xyxy[0])
    
    palate = np.zeros((500, 500))
    palate_3d = np.zeros((500, 500, 3))

    breadboard_image_center = pin_target.shape[1], pin_target.shape[0]
    for i in range(len(r)):
        data = r.iloc[i]
        if data.confidence > 0.5:
            p = [int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)]

            pad_area = area_padding(target, (p[0], p[1]), (p[2], p[3]), base_point[0], base_point[2], expand_to = 350)
            area = pad_area.copy()
            color_area = area.copy()

            cv2.rectangle(target, (p[0], p[1]), (p[2], p[3]), (0, 255, 0), 5)

            search_map = None
            cols = None
            x_s = None

            if p[2] > breadboard_image_center[0]/2:
                search_map = body_pinmap.loc[:, "F":"J"]
                cols = ["F", "G", "H", "I", "J"]  
                
                x_s = [int(breadboard_image_center[0]/2)]
                x_s += [int(search_map[col, "x"].median()) for col in cols]

            else:
                search_map = body_pinmap.loc[:, "A":"E"]
                cols = ["A", "B", "C", "D", "E"]

                x_s = [int(breadboard_image_center[0]/2)]
                x_s += [int(search_map[col, "x"].median()) for col in cols]

            x_s = np.array(x_s, np.uint32)
            cols = np.array(cols)
            idx = np.where((x_s > p[0]) & (x_s < p[2]))

            try:
                x_res = cols[idx]

                if len(x_res):
                    cols = cols[idx].tolist()

                    new_x_s = x_s[idx]
                    
                    x_s_border = [int(median([new_x_s[i], new_x_s[i+1]])) for i in range(len(new_x_s)) if i != len(new_x_s) - 1]

                    rows = np.arange(30, dtype=np.uint32)
                    y_s = [PADDING]
                    y_s += search_map[cols[0], "y"].tolist()

                    y_s = np.array(y_s, np.uint32)
                    idx = np.where((y_s > p[1]) & (y_s < p[3]))[0]
                    cols = np.array(idx - 1, np.str0).tolist()

                    new_y_s = y_s[idx]
                    y_s_border = [int(median([new_y_s[i], new_y_s[i+1]])) for i in range(len(new_y_s)) if i != len(new_y_s) - 1]
                    table_idx = [x + r for x in x_res for r in cols] 

                    # border 만들고 여기다 drawing
                    for x_border in x_s_border:
                        x_border = x_border - p[0]
                        cv2.line(pad_area, (x_border, 0), (x_border, 350), (255, 10, 10), 5)

                    for y_border in y_s_border:
                        y_border = y_border - p[1]
                        cv2.line(pad_area, (0, y_border), (350, y_border), (53, 100, 20), 5)

            except IndexError as e:
                pass
                # print(p)

            # if -add_height + p[1] > 

            cv2.imshow(f"pad_area_{i}", pad_area)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            # area = cv2.adaptiveThreshold(area, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)
            _, area = cv2.threshold(area, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            # area = cv2.morphologyEx(area, cv2.MORPH_GRADIENT, np.ones((3, 2), np.uint8), iterations=1)
            area = cv2.morphologyEx(area, cv2.MORPH_OPEN, kernel, iterations=7)
            # area = cv2.morphologyEx(area, cv2.MORPH_ERODE, kernel, iterations=3)

            # templat = 255 * np.ones((40, 40), np.uint8)
            # templat = np.pad(templat, (5, 5), 'constant', constant_values=0)
            # res = cv2.matchTemplate(area, templat, cv2.TM_CCOEFF_NORMED)
            # thresh = 0.7
            # box_loc = np.where(res >= thresh)

            # for box in zip(*box_loc[::-1]):
            #     startX, startY = box
            #     endY, endX = templat.shape 
            #     endX += startX + 5
            #     endY += startY + 5
            #     cv2.rectangle(color_area, (startX, startY), (endX, endY), (0, 0, 255), 2)


            contours, hierarchy = cv2.findContours(area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                min_rect = cv2.minAreaRect(contour)
                bbox = np.uint0(cv2.boxPoints(min_rect))
                cv2.drawContours(color_area, [bbox], 0, (36, 142, 183), 3)
                # ep3 = 0.05 * cv2.arcLength(contour, True)
                # approx3 = cv2.approxPolyDP(contour, ep3, True)
                # cv2.drawContours(color_area, [approx3], -1, (0, 255, 0), 3)

                # print(ep3, len(approx3))

            area_coords = np.array(np.where(area == 0)).reshape(-1, 2)

            kmeans = KMeans(n_clusters=1, tol=0.01)
            kmeans.fit(area_coords)

            centroid = kmeans.cluster_centers_

            for coord in centroid:
                center = int(coord[0]), int(coord[1])
                cv2.circle(color_area, center, 15, (255, 255, 0), cv2.FILLED)
                cv2.circle(area, center, 15, (0, 0, 0), cv2.FILLED)
            # 255로 크기 맞춤
            # h, w
            # to_x1 = 0; to_x2 = 0; to_y1 = 0; to_y2 = 0;
            # if int((255 - area.shape[0])/2) % 2 == 1:
            #     to_x1 = int((255 - area.shape[0])/2)
            #     to_x2 = to_x1 + 1
            # else:
            #     to_x1 = to_x2 = int((255 - area.shape[0])/2)
                
            # if int((255 - area.shape[1])/2) % 2 == 1:
            #     to_y1 = int((255 - area.shape[1])/2)
            #     to_y2 = to_y1 + 1
            # else:
            #     to_y1 = to_y2 = int((255 - area.shape[1])/2)

            # print((to_x1, to_x2), (to_y1, to_y2))

            # area = np.pad(area, ((to_x1, to_x2), (to_y1, to_y2)), 'constant', constant_values=255)
            # 256을 넘어가는 영역이 인식 됨..

            # area = cv2.resize(area, (500, 500))
            # color_area = cv2.resize(color_area, (500, 500))
            cv2.imshow(f"b_{i}", color_area)

            # palate = np.hstack((palate, area))
            # palate_3d = np.hstack((palate_3d, color_area))

            # view = area.view(dtype=np.uint8, type=np.matrix)
            # np.savetxt(f"b_{i}.txt", view, fmt="%3d", delimiter=" ")
        # cv2.imshow(f"end_point_res", palate)
        # cv2.imshow(f"end_point_3d", palate_3d)

    cv2.imshow("res", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()