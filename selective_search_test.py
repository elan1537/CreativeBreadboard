import selectivesearch
import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np

GT = [
        [ 1000, 552 ],
        [ 470, 2925 ],
        [ 1957, 3434 ],
        [ 2602, 895 ]
]

gt = (470, 552, 2602, 3434)

def compute_iou(cand_box, gt_box):
    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.minimum(cand_box[2], gt_box[2])
    y2 = np.minimum(cand_box[3], gt_box[3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area

    return intersection / union

img = cv2.imread("./images/Circuits/220428/Circuit-15.220428.jpg", cv2.COLOR_BGR2RGB)
print(f'img_shape: {img.shape}, size: {img.shape[0] * img.shape[1]}')

img_gt = img.copy()
cv2.rectangle(img_gt, (gt[0], gt[1]), (gt[2], gt[3]), color=(255, 0, 0), thickness=5)
cv2.imshow('img_gt', img_gt)

min_sz = int(img.shape[0] * img.shape[1] / 15)

_, regions = selectivesearch.selective_search(img, scale=100, min_size=min_sz)
print(type(regions), len(regions))
print(regions)

regions = [{'rect': (0, 0, 971, 2227), 'size': 1249939, 'labels': [0.0]}, {'rect': (317, 0, 2706, 932), 'size': 1757266, 'labels': [1.0]}, {'rect': (383, 556, 1426, 2631), 'size': 2113532, 'labels': [2.0]}, {'rect': (1071, 662, 1952, 3369), 'size': 2946497, 'labels': [3.0]}, {'rect': (1225, 718, 1434, 2719), 'size': 2220101, 'labels': [4.0]}, {'rect': (0, 771, 1293, 3260), 'size': 1905433, 'labels': [5.0]}, {'rect': (0, 0, 3023, 2227), 'size': 3007205, 'labels': [0.0, 1.0]}, {'rect': (383, 556, 2276, 2881), 'size': 4333633, 'labels': [2.0, 4.0]}, {'rect': (0, 0, 3023, 4031), 'size': 5953702, 'labels': [0.0, 1.0, 3.0]}, {'rect': (0, 0, 3023, 4031), 'size': 7859135, 'labels': [0.0, 1.0, 3.0, 5.0]}, {'rect': (0, 0, 3023, 4031), 'size': 12192768, 'labels': [0.0, 1.0, 3.0, 5.0, 2.0, 4.0]}]

cand_rects = [cand['rect'] for cand in regions]

# for rect in cand_rects:
#     R = int(random.random() * 255)
#     G = int(random.random() * 255)
#     B = int(random.random() * 255)

#     left = rect[0]
#     top = rect[1]
#     right = left + rect[2]
#     bottom = top + rect[3]

#     print(left, top, right, bottom)
#     cv2.rectangle(img, (left, top), (right, bottom), color=(B, G, R), thickness=10)

for idx, rect in enumerate(cand_rects):
    rect = list(rect)
    rect[2] += rect[0]
    rect[3] += rect[1] 

    iou = compute_iou(rect, gt)
    print(f'index: {idx}, iou: {iou}')

    if iou > 0.1:
        R = int(random.random() * 255)
        G = int(random.random() * 255)
        B = int(random.random() * 255)

        left = rect[0]
        top = rect[1]
        right = left + rect[2]
        bottom = top + rect[3]

        print(left, top, right, bottom)
        cv2.rectangle(img, (left, top), (right, bottom), color=(B, G, R), thickness=10)


cv2.imshow('img2', img)
cv2.waitKey(0)
cv2.destroyAllWindows()