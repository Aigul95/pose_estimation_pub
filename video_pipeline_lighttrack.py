import json
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import PIL.Image
from person_tracker_ak import Tracker
from pose_2d_inferences import Pose2DInference
from pose_3d_inference import Pose3DInference
from progress.bar import IncrementalBar
from s3fd_inference import S3FDInference
from utils import find_color_scalar

warnings.filterwarnings("ignore")
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

THRESH_DETECT_PERSON = 0.9
THRESH_IOU = 0.2
TIME_NON_PERSON = 1
img_Y = 480
FACE_PROB_THRESH = 0.5
COLORS = [
    "purple",
    "yellow",
    "blue",
    "green",
    "red",
    "skyblue",
    "navyblue",
    "azure",
    "slate",
    "chocolate",
    "olive",
    "orange",
    "orchid",
]


class tracking_processing:
    def __init__(self):
        self.PATH_TO_2D_POSE_ESTIMATOR_224 = (
            "models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
        )
        self.PATH_TO_POSE_TOPOLOGY_224 = "models/human_pose_rmppe_224.json"
        self.PATH_TO_2D_POSE_ESTIMATOR_368 = "models/pose_model.pth"
        self.PATH_TO_POSE_TOPOLOGY_368 = "models/human_pose_rmppe.json"
        self.PATH_TO_2D_POSE_ESTIMATOR_135P = "models/SNWBPE.pth"
        self.PATH_TO_POSE_TOPOLOGY_135P = "models/human_pose_snwbpe.json"
        self.PATH_TO_3D_POSE_ESTIMATOR = "models/pretrained_h36m_detectron_coco.bin"
        pass

    def init_models(self, pose_model_type="rmppe"):
        if pose_model_type == "rmppe":
            self.pose_2d_estimator = Pose2DInference(
                self.PATH_TO_2D_POSE_ESTIMATOR_368,
                self.PATH_TO_POSE_TOPOLOGY_368,
                pose_model_type,
                link_thresh=0.2,
            )
        elif pose_model_type == "snwbpe":
            self.pose_2d_estimator = Pose2DInference(
                self.PATH_TO_2D_POSE_ESTIMATOR_135P,
                self.PATH_TO_POSE_TOPOLOGY_135P,
                pose_model_type,
            )
        elif pose_model_type == "rmppe_224":
            self.pose_2d_estimator = Pose2DInference(
                self.PATH_TO_2D_POSE_ESTIMATOR_224,
                self.PATH_TO_POSE_TOPOLOGY_224,
                pose_model_type,
                0,
                0.5,
            )
        self.pose_3d_estimator = Pose3DInference(self.PATH_TO_3D_POSE_ESTIMATOR, pose_model_type)
        self.pose_model_type = pose_model_type
        self.pose_track = Pose_Matcher()
        pass

    def img_blured_face(self, img):
        output_face_det = self.face_detector(img)
        indices_face = np.where(output_face_det["detection_scores"] > FACE_PROB_THRESH)[0]
        if len(indices_face) > 0:
            for box in output_face_det["detection_boxes"][indices_face]:
                xmin, ymin, xmax, ymax = box
                xmin = int(xmin * img.shape[1])
                ymin = int(ymin * img.shape[0])
                xmax = int(xmax * img.shape[1])
                ymax = int(ymax * img.shape[0])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                img[ymin:ymax, xmin:xmax] = cv2.GaussianBlur(
                    img[ymin:ymax, xmin:xmax], (51, 51), 200
                )
        return img

    def img_with_labels(self, img, person_tracker):
        for track_idx, track in enumerate(person_tracker.tracks):
            color_id = track.id % len(COLORS)
            img = cv2.rectangle(
                img,
                (track.bbox[0], track.bbox[1]),
                (track.bbox[2], track.bbox[3]),
                find_color_scalar(COLORS[color_id]),
                2,
            )
            cv2.putText(
                img,
                str(track.id),
                (track.bbox[0], track.bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                3,
                find_color_scalar(COLORS[color_id]),
                2,
                cv2.LINE_AA,
            )
            img = self.pose_2d_estimator.visualize(img, track.kps, COLORS[color_id])
        return img

    def video_processing(self, video_path, output_folder, draw_labels=False, blur_face=False):

        self.video_capturer = cv2.VideoCapture(video_path[0])
        width = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.person_tracker = Tracker(
            width,
            height,
            self.pose_track,
            threshold=0.5,
            tracks_length=2,
            downgrade=-5,
            dist="pose",
        )

        if draw_labels:
            width = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(
                os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(video_path[0]))[0]
                    + "_pose_track_{}.avi".format(self.pose_model_type),
                ),
                cv2.VideoWriter_fourcc(*"XVID"),
                25,
                (width, height),
            )
        self.video_capturer.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        self.amount_of_frames = self.video_capturer.get(cv2.CAP_PROP_FRAME_COUNT)
        bar = IncrementalBar("video", max=self.amount_of_frames)
        while True:
            ret, img = self.video_capturer.read()
            if ret:
                bar.next()
                img_vis = img.copy()
                obj_dict = self.pose_2d_estimator(img)
                obj_dict_list = []
                for obj in obj_dict:
                    obj_dict_list.append(obj_dict[obj])
                self.person_tracker.update(obj_dict_list)
                for i in range(len(self.person_tracker.tracks)):
                    # assign to object predicted kps without interpolation
                    _, self.person_tracker.tracks[i].kps_3d = self.pose_3d_estimator(
                        self.person_tracker.tracks[i].kps, img.shape[1], img.shape[0]
                    )
                if blur_face:
                    img_vis = self.img_blured_face(img_vis)
                if draw_labels:
                    img_vis = draw_labels(img, self.person_tracker)
                self.out.write(img_vis)
            else:
                break
        bar.finish()
        pass
