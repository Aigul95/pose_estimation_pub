import uuid
from collections import namedtuple

import cv2
import numpy as np
from pose_track_inference import *
from scipy.optimize import linear_sum_assignment

State = namedtuple("State", ["bbox", "bbox_rel", "kps"])


def revert_iou(bbox_prev, bbox):
    xa = max(bbox_prev[0], bbox[0])
    ya = max(bbox_prev[1], bbox[1])
    xb = min(bbox_prev[2], bbox[2])
    yb = min(bbox_prev[3], bbox[3])
    interArea = abs(max((xb - xa, 0)) * max((yb - ya), 0))
    if interArea == 0:
        return 1
    boxAArea = abs((bbox_prev[2] - bbox_prev[0]) * (bbox_prev[3] - bbox_prev[1]))
    boxBArea = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    r_iou = float(boxAArea + boxBArea - 2 * interArea) / float(boxAArea + boxBArea - interArea)
    return r_iou


def interpolate_pose_2d_path_track(kps2d_path):
    """
    :param: kps2d_path: list of lists of 18 kps
    :return: list of 18 kps averaged on path of track
    """
    # TODO fill np.array of kps for several pose Px18x2
    kps_path = kps2d_path.copy()
    kps_path_np = []
    for kps in kps_path:
        for i in np.where(np.array(kps) == None)[0]:
            kps[i] = [np.nan, np.nan]
        kps_path_np.append(kps)
    kps_path_np = np.array(kps_path_np)
    # TODO interpolate kps for same kps in different state of the track
    indices = np.arange(len(kps_path_np))
    for i in range(kps_path_np.shape[1]):
        mask = ~np.isnan(kps_path_np[:, i, 0])
        # TODO don't interpolate kps which don't have any gt in path of the track
        if np.any(mask):
            for j in range(kps_path_np.shape[2]):
                kps_path_np[:, i, j] = np.interp(indices, indices[mask], kps_path_np[mask, i, j])
    return kps_path_np


class Track:
    def __init__(self, state, max_length, min_downgrade):
        self._state = state
        self.uuid = str(uuid.uuid4())
        self.max_length = max_length
        self._path = [state]
        # TODO invisible count
        self._downgrade = 0
        self.min_downgrade = min_downgrade
        self.dead = False
        self.id = None

    def update(self, state):
        # TODO if there is a refreshment of the track, then will update = > invisible count will set to zero
        if self._downgrade < 0:
            self._downgrade = 0
        self._state = state
        self._path.append(state)
        # TODO if length of the track is more than max_length, then will clear the first element
        if len(self._path) >= self.max_length + 1:
            self._path.pop(0)

    def downgrade(self, img_width, img_height):
        self._downgrade -= 1
        if len(self._path) > 1:
            recent_path_length = min(10, len(self._path))
            box_end = self._path[-1].bbox
            box_start = self._path[-2].bbox
            kps = self._path[-1].kps.copy()

            point_end = np.array([(box_end[0] + box_end[2]) / 2, (box_end[1] + box_end[3]) / 2])
            point_start = np.array(
                [(box_start[0] + box_start[2]) / 2, (box_start[1] + box_start[3]) / 2]
            )

            recent_path_speed = point_end - point_start
            if recent_path_length == 1:
                recent_path_speed = np.zeros((2,), dtype=np.int32)
            else:
                recent_path_speed = recent_path_speed / (recent_path_length - 1)

            s_point = point_end + recent_path_speed
            box_end_w, box_end_h = box_end[2] - box_end[0], box_end[3] - box_end[1]
            s_box = np.array(
                [
                    s_point[0] - box_end_w / 2,
                    s_point[1] - box_end_h / 2,
                    s_point[0] + box_end_w / 2,
                    s_point[1] + box_end_h / 2,
                ]
            )
            bbox_rel = [
                s_box[0] / img_width,
                s_box[1] / img_height,
                s_box[2] / img_width,
                s_box[3] / img_height,
            ]
            synthetic_state = State(bbox=s_box, bbox_rel=bbox_rel, kps=kps)
            self._path.append(synthetic_state)
            self._point = synthetic_state

            if len(self._path) == self.max_length + 1:
                self._path.pop(0)
        pass

    @property
    def bbox(self):
        return self._state.bbox

    @property
    def bbox_rel(self):
        return self._state.bbox_rel

    @property
    def kps(self):
        return self._state.kps

    @property
    def uid(self):
        return self.uuid

    @property
    def path(self):
        return self._path

    @property
    def is_dead(self):
        # assumed y, x format
        if self._downgrade == self.min_downgrade:
            return True
        else:
            return False

    def mean_pose(self):
        if len(self._path) > 1:
            kps_path = []
            for state in self._path:
                kps = state.kps.copy()
                kps_path.append(kps)
            kps_interp = interpolate_pose_2d_path_track(kps_path)
            kps_average = np.mean(kps_interp, axis=0)
            nan_indices = np.where(np.isnan(kps_interp[:, 0]))[0]
            kps_average = kps_average.tolist()
            for idx in nan_indices:
                kps_average[idx] = None
            return kps_average
        else:
            return self._path[0].kps


class Tracker(object):
    def __init__(
        self,
        img_width,
        img_height,
        pose_track_model,
        threshold=0.2,
        tracks_length=10,
        downgrade=-5,
        dist="pose",
    ):
        self.tracks = []
        self._threshold = threshold
        self._tracks_length = tracks_length
        self._downgrade = downgrade
        self._img_width = img_width
        self._img_height = img_height
        self.pose_track_model = pose_track_model
        self.cnt_id = 0
        self.dist_type = dist

    def kill_tracks(self):
        for ind, track in enumerate(self.tracks):
            if track.dead():
                self.tracks.pop(ind)

    def _match(self, kps_det):

        # TODO get cost matrix of dists from inference SGCN (lighttrack) or of dists IOU
        cost = np.zeros((len(self.tracks), len(kps_det)))
        if self.dist_type == "pose":
            for i, track in enumerate(self.tracks):
                for j, kps in enumerate(kps_det):
                    mean_kps = track.mean_pose()
                    # _, dist = get_pose_matching_score(track.kps, kps, self.pose_track_model, self._img_width, self._img_height)
                    _, dist = get_pose_matching_score(
                        mean_kps, kps, self.pose_track_model, self._img_width, self._img_height
                    )
                    cost[i, j] = dist
            row_ind, col_ind = linear_sum_assignment(cost)
            # TODO filter by the threshold
            matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost[i, j] <= self._threshold]

            matched_p = [m[0] for m in matches]
            matched_d = [m[1] for m in matches]

            unmatched_p = [i for i in range(len(self.tracks)) if i not in matched_p]
            unmatched_d = [i for i in range(len(kps_det)) if i not in matched_d]
        else:
            for i, track in enumerate(self.tracks):
                for j, kps in enumerate(kps_det):
                    bbox_new = get_enlarged_bbox_from_keypoints(kps)
                    r_iou = revert_iou(bbox_new, track.bbox_rel)
                    cost[i, j] = r_iou
            row_ind, col_ind = linear_sum_assignment(cost)
            # TODO filter by the threshold
            matches = [(i, j) for i, j in zip(row_ind, col_ind)]
            # if cost[i, j] >= self._threshold]

            matched_p = [m[0] for m in matches]
            matched_d = [m[1] for m in matches]

            unmatched_p = [i for i in range(len(self.tracks)) if i not in matched_p]
            unmatched_d = [i for i in range(len(kps_det)) if i not in matched_d]

        return matches, unmatched_p, unmatched_d

    def update(self, kps_det):
        if len(self.tracks) > 0:
            matches, unmatched_tracks, unmatched_kps_det = self._match(kps_det)
            for m in matches:
                bbox_rel = get_enlarged_bbox_from_keypoints(kps_det[m[1]])
                bbox = [
                    int(bbox_rel[0] * self._img_width),
                    int(bbox_rel[1] * self._img_height),
                    int(bbox_rel[2] * self._img_width),
                    int(bbox_rel[3] * self._img_height),
                ]
                state = State(bbox=bbox, bbox_rel=bbox_rel, kps=kps_det[m[1]])
                self.tracks[m[0]].update(state)
            # TODO revert sequence of unmatched tracks to downgrade and to clear them correctly
            for ut in sorted(unmatched_tracks, reverse=True):
                self.tracks[ut].downgrade(self._img_width, self._img_height)
                if self.tracks[ut].is_dead:
                    self.tracks[ut].dead = True
                    self.tracks.pop(ut)
            for ud in unmatched_kps_det:
                bbox_rel = get_enlarged_bbox_from_keypoints(kps_det[ud])
                bbox = [
                    int(bbox_rel[0] * self._img_width),
                    int(bbox_rel[1] * self._img_height),
                    int(bbox_rel[2] * self._img_width),
                    int(bbox_rel[3] * self._img_height),
                ]
                state = State(bbox=bbox, bbox_rel=bbox_rel, kps=kps_det[ud])
                track = Track(state, self._tracks_length, self._downgrade)
                self.cnt_id += 1
                track.id = self.cnt_id
                self.tracks.append(track)
        else:
            for i in range(len(kps_det)):
                bbox_rel = get_enlarged_bbox_from_keypoints(kps_det[i])
                bbox = [
                    int(bbox_rel[0] * self._img_width),
                    int(bbox_rel[1] * self._img_height),
                    int(bbox_rel[2] * self._img_width),
                    int(bbox_rel[3] * self._img_height),
                ]
                state = State(bbox=bbox, bbox_rel=bbox_rel, kps=kps_det[i])
                track = Track(state, self._tracks_length, self._downgrade)
                self.cnt_id += 1
                track.id = self.cnt_id
                self.tracks.append(track)
        pass
