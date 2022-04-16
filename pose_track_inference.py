import argparse
import sys
import time

import cv2
import graph.torchlight
import numpy as np
import torch
from graph.gcn_utils.gcn_model import Model
from graph.gcn_utils.io import IO
from graph.gcn_utils.processor_siamese_gcn import SGCN_Processor

ORDER_SGCN = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0]


class Pose_Matcher(SGCN_Processor):
    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        return

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=False,
            parents=[parent_parser],
            description="Graph Convolution Network for Pose Matching",
        )
        # parser.set_defaults(config='config/inference.yaml')
        parser.set_defaults(config="graph/config/inference.yaml")
        return parser

    def inference(self, data_1, data_2):
        self.model.eval()

        with torch.no_grad():
            data_1 = torch.from_numpy(data_1)
            data_1 = data_1.unsqueeze(0)
            data_1 = data_1.float().to(self.dev)

            data_2 = torch.from_numpy(data_2)
            data_2 = data_2.unsqueeze(0)
            data_2 = data_2.float().to(self.dev)

            feature_1, feature_2 = self.model.forward(data_1, data_2)

        # euclidian distance
        diff = feature_1 - feature_2
        dist_sq = torch.sum(pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        margin = 0.2
        distance = dist.data.cpu().numpy()[0]
        # print("_____ Pose Matching: [dist: {:04.2f}]". format(distance))
        if dist >= margin:
            return False, distance  # Do not match
        else:
            return True, distance  # Match


def interpolate_pose_2d(kps2d):
    kps = kps2d.copy()
    for i in np.where(np.array(kps) == None)[0]:
        kps[i] = [np.nan, np.nan]
    kps = np.array(kps)
    mask = ~np.isnan(kps[:, 0])
    indices = np.arange(len(kps))
    for j in range(2):
        kps[:, j] = np.interp(indices, indices[mask], kps[mask, j])
    return kps


def kps_from_rmppe2sgcn_format(kps):
    """
    :param kps: list of 18 kps in rmppe format by human_pose_rmppe topology
    :return: list of 15 kps in sgcn format by human_pose_sgcn topology
    """
    kps_arr = np.array(kps)
    if len(kps_arr.shape) == 1:
        kps2d = np.take(kps_arr, ORDER_SGCN).tolist()
    else:
        kps2d = kps_arr[ORDER_SGCN, :].tolist()
    # extrapolate kp for top head (coco kps don't have top head kp)
    kps_arr = interpolate_pose_2d(kps2d)
    kp_th = np.array([2 * kps_arr[13, 0] - kps_arr[12, 0], 2 * kps_arr[13, 1] - kps_arr[12, 1]])
    kps_arr = np.concatenate((kps_arr, kp_th[np.newaxis, :]))
    return kps_arr.tolist()


def enlarge_bbox(bbox, scale):
    assert scale > 0
    min_x, min_y, max_x, max_y = bbox
    margin_x = 0.5 * scale * (max_x - min_x)
    margin_y = 0.5 * scale * (max_y - min_y)
    if margin_x < 0:
        margin_x = 2
    if margin_y < 0:
        margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x = 0
        max_x = 2
        min_y = 0
        max_y = 2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def get_enlarged_bbox_from_keypoints(keypoints):
    """

    :param keypoints: list of kps 15*[(x, y)] by topology human_pose_sgcn
    :return: enlarged bbox list [top, left, right, bottom]
    """
    indices_is_none = np.where(np.array(keypoints) == None)[0]
    if indices_is_none.shape[0] > 0:
        keypoints = [keypoints[i] for i in range(len(keypoints)) if i not in indices_is_none]
    keypoints = np.array(keypoints)
    min_x = np.min(keypoints[:, 0])
    min_y = np.min(keypoints[:, 1])
    max_x = np.max(keypoints[:, 0])
    max_y = np.max(keypoints[:, 1])

    scale = 0.2  # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    # bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox


def keypoints_to_graph(keypoints, bbox):
    """
    :param keypoints: list of kps 15*[(x, y)] by topology human_pose_sgcn
    :param bbox: list [top, left, right, bottom]
    :return: graph array 15x2
    """
    x1, y1, _, _ = bbox
    # flag_pass_check = True
    keypoints = np.array(keypoints)
    graph = np.zeros(keypoints.shape)
    graph[:, 0] = keypoints[:, 0] - x1
    graph[:, 1] = keypoints[:, 1] - y1
    # graph = graph.astype(np.int64)
    return graph  # , flag_pass_check


def graph_pair_to_data(sample_graph_pair):
    data_numpy_pair = []
    for siamese_id in range(2):
        # fill data_numpy
        data_numpy = np.zeros((2, 1, 15, 1))

        pose = sample_graph_pair[:][siamese_id]
        data_numpy[0, 0, :, 0] = [x[0] for x in pose]
        data_numpy[1, 0, :, 0] = [x[1] for x in pose]
        data_numpy_pair.append(data_numpy)
    return data_numpy_pair[0], data_numpy_pair[1]


def get_pose_matching_score(keypoints_A, keypoints_B, pose_matcher, img_width, img_height):

    keypoints_A = kps_from_rmppe2sgcn_format(keypoints_A)
    keypoints_B = kps_from_rmppe2sgcn_format(keypoints_B)

    bbox_A = get_enlarged_bbox_from_keypoints(keypoints_A)
    bbox_B = get_enlarged_bbox_from_keypoints(keypoints_B)

    graph_A = keypoints_to_graph(keypoints_A, bbox_A)

    graph_B = keypoints_to_graph(keypoints_B, bbox_B)

    graph_A[:, 0] *= img_width
    graph_A[:, 1] *= img_height
    graph_B[:, 0] *= img_width
    graph_B[:, 1] *= img_height

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    flag_match, dist = pose_matcher.inference(data_A, data_B)
    return flag_match, dist
