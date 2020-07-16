
from __future__ import print_function
import copy
import numpy as np
from numba import jit
from scipy.spatial import ConvexHull
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
from shapely.geometry import Polygon


@jit
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@jit
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


@jit
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)

def box_overlap(bbox_a, bbox_b):
    """
    bbox_a : 3 X 8
    bbox_b : 3 X 8
    """
    amax = np.max(bbox_a[0:2, 0:4], axis=1)
    bmax = np.max(bbox_b[0:2, 0:4], axis=1)
    amin = np.min(bbox_a[0:2, 0:4], axis=1)
    bmin = np.min(bbox_b[0:2, 0:4], axis=1)
    min_max = np.min(np.stack(amax, bmax), axis=0)
    max_min = np.max(np.stack(amin, bmin), axis=0)
    if min_max[0] <= max_min[0] or min_max[0] <= max_min[0]:
        return 0.0
    iarea = (min_max[0] - max_min[0]) * (min_max[1] - max_min[1])
    aarea = (amax[0] - amin[0]) * (amax[1] - amin[1])
    barea = (bmax[0] - bmin[0]) * (bmax[1] - bmin[1])
    iou = iarea / (aarea + barea - iarea)
    return iou

def boxoverlap(self, a, b, criterion="union"):
    """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
    """

    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    w = x2 - x1
    h = y2 - y1

    if w <= 0. or h <= 0.:
        return 0.
    inter = w * h
    aarea = (a.x2 - a.x1) * (a.y2 - a.y1)
    barea = (b.x2 - b.x1) * (b.y2 - b.y1)
    # intersection over union overlap
    if criterion.lower() == "union":
        o = inter / float(aarea + barea - inter)
    elif criterion.lower() == "a":
        o = float(inter) / float(aarea)
    else:
        raise TypeError("Unkown type for criterion")
    return o



# def lwh_to_box(l, w, h):
#     box = np.array([
#         [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
#         [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
#         [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
#     ])
#     return box
def bbox_adjacency_2d(box_a, box_b):
    dist = np.linalg.norm(box_a[0:2]-box_b[0:2])
    max_size_a = np.max(box_a[4:6])
    max_size_b = np.max(box_b[4:6])
    return dist < max_size_a + max_size_b


def bbox_iou2d(box_a, box_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        return 0
    union = (xy_poly_a.area + xy_poly_b.area - xy_intersection)
    if union == 0:
        return 0
    return xy_intersection / union


def bbox_iou3d(box_a, box_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # height (Z) overlap
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0.

    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        return 0.

    a_vol = (max_h_a - min_h_a) * xy_poly_a.area
    b_vol = (max_h_b - min_h_b) * xy_poly_b.area
    intersection_vol = z_intersection * xy_intersection

    return intersection_vol / (a_vol + b_vol - intersection_vol)


@jit
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


@jit
def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def corners(bbox3d):
    """
    Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            5 -------- 4
           /|         /|
          6 -------- 7 .
          | |        | |
          . 1 -------- 0
          |/         |/       z|__y
          2 -------- 3         /x

    Returns the bounding box corners.

    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    # w, l, h =  * wlh_factor
    w = bbox3d[4]
    l = bbox3d[5]
    h = bbox3d[6]

    R = rotz(bbox3d[2])
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    # x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    # y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    # z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    x_corners = l / 2 * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(R, corners)

    # Translate
    corners[0, :] = corners[0, :] + bbox3d[0]
    corners[1, :] = corners[1, :] + bbox3d[1]
    corners[2, :] = corners[2, :] + bbox3d[3]

    return np.transpose(corners)



def angle_in_range(angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def diff_orientation_correction(det, trk):
    '''
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    '''
    diff = det - trk
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    diff = angle_in_range(diff)
    return diff
