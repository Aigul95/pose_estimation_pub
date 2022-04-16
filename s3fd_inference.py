import cv2
import face_not_face_s3fd.net_s3fd as net_s3fd
import numpy as np
import torch
import torch.nn.functional as F
from face_not_face_s3fd.bbox import *
from face_not_face_s3fd.face_not_face_s3fd import *
from torch.autograd import Variable

IMG_Y = 480


class S3FDInference:
    def __init__(self, path_to_model):
        self.detector_face = net_s3fd.s3fd()
        self.detector_face.load_state_dict(torch.load(path_to_model))
        self.detector_face.cuda()
        self.detector_face.eval()
        pass

    def __call__(self, img):
        if img.shape[0] != IMG_Y:
            img = cv2.resize(
                img,
                (int(img.shape[1] * IMG_Y / img.shape[0]), IMG_Y),
                interpolation=cv2.INTER_AREA,
            )
        bboxlist = self.detect_face(img)
        if bboxlist.size > 0:
            keep = nms(bboxlist, 0.3)
            if len(keep) != 0:
                bboxlist = bboxlist[keep, :]
                output = {}
                rel_boxes = np.zeros(np.array(bboxlist[:, :-1]).shape)
                rel_boxes[:, 0] = np.array(bboxlist[:, 0]) / img.shape[1]
                rel_boxes[:, 1] = np.array(bboxlist[:, 1]) / img.shape[0]
                rel_boxes[:, 2] = np.array(bboxlist[:, 2]) / img.shape[1]
                rel_boxes[:, 3] = np.array(bboxlist[:, 3]) / img.shape[0]
                #             output['detection_boxes'] = np.array(bboxlist[:, :-1])
                output["detection_boxes"] = rel_boxes
                output["detection_scores"] = np.array(bboxlist[:, 4])
                return output

    def detect_face(self, img):
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)

        img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
        BB, CC, HH, WW = img.size()
        olist = self.detector_face(img)

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2])
        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                #             bboxlist.append([x1 / img.shape[3], y1 / img.shape[2], x2 / img.shape[3], y2 / img.shape[2], score])
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))
        return bboxlist
