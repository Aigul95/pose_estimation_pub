"""
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Nov 2nd, 2018

    Load keypoints from existing openSVAI data format
    and turn these keypoints into Graph structure for GCN

    N: # of batch_size
    M: # of instances within a frame (which is # of human candidates)
    V: # of graph nodes (which is 15)
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("../../"))

from utils_io_folder import *
from utils_json import *


def load_data_for_gcn(dataset_str, dataset_split_str):
    if dataset_str == "posetrack_18":
        if dataset_split_str == "train":
            json_folder_path = (
                "/export/guanghan/Data_2018_lighttrack/posetrack_data/gcn_openSVAI/train"
            )
        elif dataset_split_str == "val":
            json_folder_path = (
                "/export/guanghan/Data_2018_lighttrack/posetrack_data/gcn_openSVAI/val"
            )
        elif dataset_split_str == "test":
            json_folder_path = (
                "/export/guanghan/Data_2018_lighttrack/posetrack_data/gcn_openSVAI/val"
            )

        json_file_paths = get_immediate_childfile_paths(json_folder_path)

        graph_pair_list_all = []
        for json_file_path in json_file_paths:
            graph_pair_list = load_graph_pairs_from_json(json_file_path)
            graph_pair_list_all.extend(graph_pair_list)

    return graph_pair_list_all


def load_graph_pairs_from_json(json_file_path):
    python_data = read_json_from_file(json_file_path)
    num_imgs = len(python_data)

    track_id_dict = {}
    for track_id in range(100):
        track_id_dict[track_id] = []

    max_track_id = -1
    for img_id in range(num_imgs):
        image_id = python_data[img_id]["image"]["id"]
        candidates = python_data[img_id]["candidates"]

        num_candidates = len(candidates)
        for candidate_id in range(num_candidates):
            candidate = candidates[candidate_id]
            track_id = candidate["track_id"]
            keypoints = candidate["pose_keypoints_2d"]
            bbox = candidate["det_bbox"]

            if track_id > max_track_id:
                max_track_id = track_id

            candidate_dict = {
                "track_id": track_id,
                "img_id": image_id,
                "bbox": bbox,
                "keypoints": keypoints,
            }
            track_id_dict[track_id].append(candidate_dict)

    graph_pair_list_all = []
    for track_id in range(max_track_id):
        candidate_dict_list = track_id_dict[track_id]
        candidate_dict_list_sorted = sorted(candidate_dict_list, key=lambda k: k["img_id"])

        graph_pair_list = get_graph_pairs(candidate_dict_list_sorted)
        graph_pair_list_all.extend(graph_pair_list)
    return graph_pair_list_all


def get_graph_pairs(candidate_dict_list_sorted):
    num_dicts = len(candidate_dict_list_sorted)
    graph_pair_list = []
    for dict_id in range(num_dicts - 1):
        candidate_dict_curr = candidate_dict_list_sorted[dict_id]
        candidate_dict_next = candidate_dict_list_sorted[dict_id + 1]

        if candidate_dict_next["img_id"] - candidate_dict_curr["img_id"] >= 2:
            continue

        keypoints_curr = candidate_dict_curr["keypoints"]
        keypoints_next = candidate_dict_next["keypoints"]

        bbox_curr = candidate_dict_curr["bbox"]
        bbox_next = candidate_dict_next["bbox"]

        # Only keep hard positives: same id, but no overlap in bbox
        iou_score = get_iou_score(bbox_curr, bbox_next)
        if iou_score >= 0:
            continue

        if validate_bbox(bbox_curr) is False:
            continue
        if validate_bbox(bbox_next) is False:
            continue

        graph_curr, flag_pass_check = keypoints_to_graph(keypoints_curr, bbox_curr)
        if flag_pass_check is False:
            continue

        graph_next, flag_pass_check = keypoints_to_graph(keypoints_next, bbox_next)
        if flag_pass_check is False:
            continue

        graph_pair = (graph_curr, graph_next)  # use tuple over list
        graph_pair_list.append(graph_pair)
    return graph_pair_list


def validate_bbox(bbox):
    x0, y0, w, h = bbox
    if w <= 100 or h <= 100:
        return False
    else:
        return True


def keypoints_to_graph(keypoints, bbox):
    num_elements = len(keypoints)
    num_keypoints = num_elements / 3
    assert num_keypoints == 15

    x0, y0, w, h = bbox
    flag_pass_check = True

    graph = 15 * [(0, 0)]
    for id in range(15):
        x = keypoints[3 * id] - x0
        y = keypoints[3 * id + 1] - y0

        score = keypoints[3 * id + 2]
        graph[id] = (int(x), int(y))
    return graph, flag_pass_check


def get_iou_score(bbox_gt, bbox_det):
    boxA = xywh_to_x1y1x2y2(bbox_gt)
    boxB = xywh_to_x1y1x2y2(bbox_det)

    iou_score = iou(boxA, boxB)
    return iou_score


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


python_data_example = {
    "version": "1.0",
    "image": [{"folder": "images/bonn_5sec/000342_mpii", "name": "00000001.jpg", "id": 0}],
    "candidates": [
        {
            "det_category": 1,
            "det_bbox": [300, 300, 100, 100],
            "det_score": [0.9],
            "pose_order": [1, 2, 3],
            "pose_keypoints_2d": [10, 10, 0.9, 20, 20, 0.9, 30, 30, 0.8],
            "track_id": [0],
            "track_score": [0.8],
        },
        {
            "det_category": 2,
            "det_bbox": [300, 300, 100, 100],
            "det_score": [0.1],
            "pose_order": [1, 2, 3],
            "pose_keypoints_2d": [10, 10, 0.9, 20, 20, 0.9, 30, 30, 0.8],
            "track_id": [1],
            "track_score": [0.6],
        },
    ],
}
