import warnings

import numpy as np
from common.camera import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.loss import *
from common.model import *

warnings.filterwarnings("ignore")

# joints from coco format
JOINTS_LEFT, JOINTS_RIGHT = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
# kps by side for keypoints symmetry in coco format (rmppe_224)
KPS_LEFT, KPS_RIGHT = [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]
# Dummy camera parameters (taken from Human3.6M), only for visualization purposes
AZIMUTH = 70
ORIENTATION = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
TRANSLATION = [1841.1070556640625, 4955.28466796875, 1563.4454345703125]
TEST_TIME_AUGMENTATION = True
ORDER_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10, 1]


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


class Pose3DInference:
    def __init__(self, path_to_model, pose_model_type):
        self.model_pose_3d = TemporalModel(
            17,
            2,
            17,
            filter_widths=[3, 3, 3, 3, 3],
            causal=False,
            dropout=0.25,
            channels=1024,
            dense=False,
        )
        receptive_field = self.model_pose_3d.receptive_field()
        print("INFO: Receptive field: {} frames".format(receptive_field))
        self.pad = (receptive_field - 1) // 2  # Padding on each side
        self.causal_shift = 0
        model_params = 0
        for parameter in self.model_pose_3d.parameters():
            model_params += parameter.numel()
        print("INFO: Trainable parameter count:", model_params)
        if torch.cuda.is_available():
            self.model_pose_3d = self.model_pose_3d.cuda()
        print("Loading checkpoint", path_to_model)
        checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        print("This model was trained for {} epochs".format(checkpoint["epoch"]))
        self.model_pose_3d.load_state_dict(checkpoint["model_pos"])
        self.pose_model_type = "rmppe"
        pass

    def evaluate_3d(self, test_generator, action=None, return_predictions=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        with torch.no_grad():
            self.model_pose_3d.eval()
            N = 0
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()

                # Positional model
                predicted_3d_pos = self.model_pose_3d(inputs_2d)

                # Test-time augmentation (if enabled)
                if test_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    predicted_3d_pos[1, :, JOINTS_LEFT + JOINTS_RIGHT] = predicted_3d_pos[
                        1, :, JOINTS_RIGHT + JOINTS_LEFT
                    ]
                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

                if return_predictions:
                    return predicted_3d_pos.squeeze(0).cpu().numpy()

                inputs_3d = torch.from_numpy(batch.astype("float32"))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                inputs_3d[:, :, 0] = 0
                if test_generator.augment_enabled():
                    inputs_3d = inputs_3d[:1]

                error = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_pos_scale += (
                    inputs_3d.shape[0]
                    * inputs_3d.shape[1]
                    * n_mpjpe(predicted_3d_pos, inputs_3d).item()
                )

                epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                inputs = (
                    inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                )
                predicted_3d_pos = (
                    predicted_3d_pos.cpu()
                    .numpy()
                    .reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                )

                epoch_loss_3d_pos_procrustes += (
                    inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
                )

                # Compute velocity error
                epoch_loss_3d_vel += (
                    inputs_3d.shape[0]
                    * inputs_3d.shape[1]
                    * mean_velocity_error(predicted_3d_pos, inputs)
                )

        if action is None:
            print("----------")
        else:
            print("----" + action + "----")
        e1 = (epoch_loss_3d_pos / N) * 1000
        e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
        e3 = (epoch_loss_3d_pos_scale / N) * 1000
        ev = (epoch_loss_3d_vel / N) * 1000
        print("Test time augmentation:", test_generator.augment_enabled())
        print("Protocol #1 Error (MPJPE):", e1, "mm")
        print("Protocol #2 Error (P-MPJPE):", e2, "mm")
        print("Protocol #3 Error (N-MPJPE):", e3, "mm")
        print("Velocity Error (MPJVE):", ev, "mm")
        print("----------")

        return e1, e2, e3, ev

    def __call__(self, kps, img_w, img_h):
        """
        :param kps2d: list of 18 kps
        :param img_w: image width
        :param img_h: image height
        :return: np.array (1, 17, 3) - 3D kps
        """
        if kps is None or len(kps) == 0:
            return None, None
        kps_arr = np.array(kps)
        if self.pose_model_type == "rmppe":
            if len(kps_arr.shape) == 1:
                kps2d = np.take(kps_arr, ORDER_COCO).tolist()
            else:
                kps2d = kps_arr[ORDER_COCO, :].tolist()
        kps_arr = interpolate_pose_2d(kps2d)
        if len(kps_arr.shape) == 2:
            kps_arr[:, 0] = kps_arr[:, 0] * img_w
            kps_arr[:, 1] = kps_arr[:, 1] * img_h
            kps_arr = np.array([kps_arr[:-1]])
            kps_arr[..., :2] = normalize_screen_coordinates(kps_arr[..., :2], w=img_w, h=img_h)
            gen = UnchunkedGenerator(
                None,
                None,
                [kps_arr],
                pad=self.pad,
                causal_shift=self.causal_shift,
                augment=TEST_TIME_AUGMENTATION,
                kps_left=KPS_LEFT,
                kps_right=KPS_RIGHT,
                joints_left=JOINTS_LEFT,
                joints_right=JOINTS_RIGHT,
            )
            prediction = self.evaluate_3d(gen, return_predictions=True)
            rot = ORIENTATION
            prediction = prediction.astype("double")
            prediction = camera_to_world(prediction, R=rot, t=0)
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            kps_3d = []
            for i in range(17):
                if kps2d[i] is not None:
                    peak = prediction[0][i]
                    x = float(peak[0])
                    y = float(peak[1])
                    z = float(peak[2])
                    kps_3d.append([x, y, z])
                else:
                    kps_3d.append(None)
            #
            return prediction, kps_3d
        else:
            return None, None
