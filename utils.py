import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn.metrics.pairwise import paired_distances


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


def compute_similarity_transform(mat1, mat2, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Args
    X: array NxM of targets, with N number of points and M point dimensionality
    Y: array NxM of inputs
    compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
    d: squared error after transformation
    Z: transformed Y
    T: computed rotation
    b: scaling
    c: translation
    """
    co_indices = np.intersect1d(
        np.unique(np.where(mat1 != np.array(None))[0]),
        np.unique(np.where(mat2 != np.array(None))[0]),
    )
    # print(co_indices)
    if len(co_indices) > 0:
        if len(np.array(mat1).shape) > 1:
            mat1_tr = np.array(mat1)[co_indices, :]
        else:
            mat1_tr = np.take(mat1, co_indices)
        mat1_tr = np.array([x for x in mat1_tr])
        if len(np.array(mat2).shape) > 1:
            mat2 = np.array(mat2)[co_indices, :]
        else:
            mat2 = np.take(mat2, co_indices)
        mat2 = np.array([x for x in mat2])
        X = mat1_tr
        #     X = mat1
        Y = mat2
        #     print(X.shape, Y.shape)

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.0).sum()
        ssY = (Y0 ** 2.0).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 = X0 / normX
        Y0 = Y0 / normY

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        # Make sure we have a rotation
        detT = np.linalg.det(T)
        V[:, -1] *= np.sign(detT)
        s[-1] *= np.sign(detT)
        T = np.dot(V, U.T)

        traceTA = s.sum()

        if compute_optimal_scale:  # Compute optimum scaling of Y.
            b = traceTA * normX / normY
            d = 1 - traceTA ** 2
            Z = normX * traceTA * np.dot(Y0, T) + muX
        else:  # If no scaling allowed
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        c = muX - b * np.dot(muY, T)

        return d, Z, T, b, c
    else:
        return None


def iou(prev, cur):

    xa = max(prev.xmin, cur.xmin)
    ya = max(prev.ymin, cur.ymin)
    xb = min(prev.xmax, cur.xmax)
    yb = min(prev.ymax, cur.ymax)
    interArea = abs(max((xb - xa, 0)) * max((yb - ya), 0))
    if interArea == 0:
        return 0
    boxAArea = abs(prev.width * prev.height)
    boxBArea = abs(cur.width * cur.height)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prokrust(mat1, mat2):
    co_indices = np.intersect1d(
        np.unique(np.where(mat1 != np.array(None))[0]),
        np.unique(np.where(mat2 != np.array(None))[0]),
    )
    if len(co_indices) > 0:
        if len(np.array(mat1).shape) > 1:
            mat1_tr = np.array(mat1)[co_indices, :]
        else:
            mat1_tr = np.take(mat1, co_indices)
        mat1_tr = np.array([x for x in mat1_tr])
        if len(np.array(mat2).shape) > 1:
            mat2 = np.array(mat2)[co_indices, :]
        else:
            mat2 = np.take(mat2, co_indices)
        mat2 = np.array([x for x in mat2])
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        Y = pad(mat1_tr)
        X = pad(mat2)
        A, res, rank, s = np.linalg.lstsq(X, Y)
        A[np.abs(A) < 1e-10] = 0
        transform = lambda x: unpad(np.dot(pad(x), A))
        mat2_tr = transform(mat2)
        dist = np.mean(paired_distances(mat1_tr, mat2_tr))
        return dist
    else:
        return None


def kps_in_bbox(kps, bbox):
    count_exist = 18 - np.where(kps == np.array(None))[0].shape[0]
    bbox = Polygon(
        [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
    )
    count_in_bbox = 0
    for kp in kps:
        if kp is not None:
            point = Point(kp[0], kp[1])
            if bbox.contains(point):
                count_in_bbox += 1
    obj_in_bbox = False
    if count_exist > 3:
        if count_in_bbox >= count_exist - 3:
            obj_in_bbox = True
    return obj_in_bbox
