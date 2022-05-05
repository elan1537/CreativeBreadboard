import torch
import pandas as pd
import numpy as np
import cv2
import json
import os

DIR = "./images/Circuits/220428/polygons"
IMG = "./images/Circuits/220428/Circuit-20.220428.jpg"
JSON = "./images/Circuits/220428/Circuit-20.220428.json"

if __name__ == "__main__":

    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    target = cv2.imread(IMG, cv2.COLOR_RGB2BGR)
    origin_shape = target.shape

    json_data = json.load(open(JSON, "r"))
    # line_label = [e for e in json_data['shapes'] if e["label"] in ["line-white-area", "line-orange-area", "line-blue-area", "line-endpoint"]]
    structure_label = [e for e in json_data["shapes"] if "structure" in e["label"]]

    json_result = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": json_data["imagePath"],
        "imageData": json_data["imageData"]
    }

    for e in structure_label:
        space = np.zeros(shape=(target.shape[0], target.shape[1], 1), dtype="uint8")
        pts = e["points"]

        for idx in range(len(pts)):
            if idx+1 == len(pts):
                break

            pt_s = (int(pts[idx][0]), int(pts[idx][1]))
            pt_e = (int(pts[idx+1][0]), int(pts[idx+1][1]))
            cv2.line(space, pt_s, pt_e, (255, 255, 255), 80)

        _, space = cv2.threshold(space, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(space, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        t = {}
        t["label"] = e["label"]
        t["points"] = []
        t["group_id"] = e["group_id"]
        t["shape_type"] = "polygon"
        t["flags"] = e["flags"]

        for contour in contours:
            p = cv2.arcLength(contour, True)
            approximatedShape = cv2.approxPolyDP(contour, 0.001 * p, True)

            for i in range(len(approximatedShape)):
                if i+1 == len(approximatedShape):
                    break

                pt_s = approximatedShape[i][0].tolist()
                t["points"].append(pt_s)

        json_result["shapes"].append(t)

    i = 20
    file_name = f"Circuit-{i}.220428.modify.json"
    with open(os.path.join(DIR, file_name), "w") as f:
        json.dump(json_result, f)