import json

import cv2
import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from parse_objects import ParseObjects
from pose_models.rtpose_vgg import get_model
from pose_models.SNWBPE import SNWBPE
from torch2trt import TRTModule
from tqdm import tqdm
from trt_pose import *

INP_SIZE = 368
MODEL_DOWNSAMPLE = 8
WIDTH = 224
HEIGHT = 224
MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()

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


def coco_category_to_topology(coco_category, mapidx=1):
    """
    Gets topology tensor from a COCO category
    mapidx: 0 - 0, 1paf layer, 1 - 1,0 paf layer
    """
    skeleton = coco_category["skeleton"]
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        if mapidx == 0:
            topology[k][0] = 2 * k
            topology[k][1] = 2 * k + 1
        elif mapidx == 1:
            topology[k][0] = 2 * k + 1
            topology[k][1] = 2 * k
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology


def pose_output_to_dict(counts, objects, peaks):
    kps = {}
    for i in range(counts[0]):
        obj = objects[0][i]
        if len(obj[obj != -1]) > 3:
            kps["obj_{}".format(i)] = []
            C = 18
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    x = float(peak[1])
                    y = float(peak[0])
                    kps["obj_{}".format(i)].append([x, y])
                else:
                    kps["obj_{}".format(i)].append(None)
    return kps


def longest_max_size(img, max_size):
    h, w, _ = img.shape
    longest_size = max(h, w)
    factor = max_size / longest_size
    return cv2.resize(img, None, fx=factor, fy=factor)


def _factor_closest(num, factor, is_ceil=True):
    num = np.ceil(float(num) / factor) if is_ceil else np.floor(float(num) / factor)
    num = int(num) * factor
    return num


def crop_with_factor(im, dest_size=None, factor=32, is_ceil=True):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = float(dest_size) / im_size_min
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    h, w, c = im.shape
    new_h = _factor_closest(h, factor=factor, is_ceil=is_ceil)
    new_w = _factor_closest(w, factor=factor, is_ceil=is_ceil)
    im_croped = np.zeros([new_h, new_w, c], dtype=im.dtype)
    im_croped[0:h, 0:w, :] = im

    return im_croped, im_scale, im.shape


def find_color_scalar(color_string):
    color_dict = {
        "purple": (255, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "skyblue": (235, 206, 135),
        "navyblue": (128, 0, 0),
        "azure": (255, 255, 240),
        "slate": (255, 0, 127),
        "chocolate": (30, 105, 210),
        "olive": (112, 255, 202),
        "orange": (0, 140, 255),
        "orchid": (255, 102, 224),
    }
    color_scalar = color_dict[color_string]
    return color_scalar


class Pose2DInference:
    def __init__(
        self,
        path_to_model="models/pose_model.pth",
        path_to_topology="models/human_pose_rmppe.json",
        pose_model_type="rmppe",
        device=0,
        link_thresh=0.1,
    ):

        with open(path_to_topology, "r") as f:
            human_pose = json.load(f)

        if pose_model_type == "rmppe":
            self.topology = coco_category_to_topology(human_pose, 1)
            self.model = get_model("vgg19")
        elif pose_model_type == "snwbpe":
            self.topology = coco_category_to_topology(human_pose, 1)
            self.model = SNWBPE()
        elif pose_model_type == "rmppe_224":
            self.topology = coco_category_to_topology(human_pose, 0)
            self.model = TRTModule()
        self.model.load_state_dict(torch.load(path_to_model))
        self.model.to(torch.device("cuda:{}".format(device)))
        if pose_model_type == "rmppe":
            self.model.float()
        self.model.eval()
        self.pose_model_type = pose_model_type
        self.parse_objects = ParseObjects(
            self.topology, cmap_threshold=0.5, link_threshold=link_thresh, line_integral_samples=7
        )
        self.device = device
        pass

    def preprocess(self, image):

        if self.pose_model_type == "rmppe":
            img_croped, im_scale, real_shape = crop_with_factor(
                image, INP_SIZE, factor=MODEL_DOWNSAMPLE, is_ceil=True
            )
            image = img_croped.astype(np.float32)
            image = image / 256.0 - 0.5
            image = image.transpose((2, 0, 1)).astype(np.float32)
            batch_images = np.expand_dims(image, 0)
            inputs = (
                torch.from_numpy(batch_images)
                .to(torch.device("cuda:{}".format(self.device)))
                .float()
            )
        elif self.pose_model_type == "snwbpe":
            input_image = longest_max_size(image, 640)
            inputs = input_image / 256.0 - 0.5
            inputs = inputs[np.newaxis, :]
            inputs = torch.from_numpy(inputs)
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(torch.device("cuda:{}".format(self.device)))
        elif self.pose_model_type == "rmppe_224":
            image = cv2.resize(image, (WIDTH, HEIGHT))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
            image = transforms.functional.to_tensor(image).to(
                torch.device("cuda:{}".format(self.device))
            )
            image.sub_(MEAN[:, None, None]).div_(STD[:, None, None])
            inputs = image[None, ...]
        return inputs

    def __call__(self, img):
        inputs = self.preprocess(img)
        with torch.no_grad():
            if self.pose_model_type == "rmppe":
                predicted_outputs, _ = self.model(inputs)
                paf, outputs = predicted_outputs[-2], predicted_outputs[-1]
            elif self.pose_model_type == "snwbpe":
                paf, outputs = self.model(inputs.float())
            elif self.pose_model_type == "rmppe_224":
                outputs, paf = self.model(inputs)
        outputs, paf = outputs.cpu(), paf.cpu()
        counts, objects, peaks = self.parse_objects(outputs, paf)
        obj_dict = pose_output_to_dict(counts, objects, peaks)
        return obj_dict

    def visualize(self, image, obj_dict, color_string="green"):
        if obj_dict is not None:
            for kp in obj_dict:
                if kp is not None:
                    x = round(float(kp[0]) * image.shape[1])
                    y = round(float(kp[1]) * image.shape[0])
                    cv2.circle(image, (x, y), 3, find_color_scalar(color_string), 2)
            for k in range(self.topology.shape[0]):
                c_a = self.topology[k][2]
                c_b = self.topology[k][3]
                if obj_dict[c_a] is not None and obj_dict[c_b] is not None:
                    peak0 = obj_dict[c_a]
                    peak1 = obj_dict[c_b]
                    x0 = round(float(peak0[0]) * image.shape[1])
                    y0 = round(float(peak0[1]) * image.shape[0])
                    x1 = round(float(peak1[0]) * image.shape[1])
                    y1 = round(float(peak1[1]) * image.shape[0])
                    cv2.line(image, (x0, y0), (x1, y1), find_color_scalar(color_string), 2)
        return image
