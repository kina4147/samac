# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from helpers import mkdir_if_missing
from scipy.spatial import ConvexHull
from scipy.linalg import block_diag

import json
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
from tqdm import tqdm

from scipy.stats import multivariate_normal as smn
from filterpy.stats import logpdf #
from filterpy.stats import multivariate_normal as fmn

from nuscenes.utils.data_classes import LidarPointCloud
from tracker import TrackerInfo, KFTracker, UKFTracker, IMMTracker, TrackerGenerator
from collections import namedtuple
from nuscenes.utils import splits
import motion_models as mm
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils.data_classes import Box
from matplotlib import pyplot as plt
from utils import angle_in_range, corners, bbox_iou2d, bbox_iou3d, bbox_adjacency_2d, adjacency_2d

from covariance import Covariance
# AB3DMOTTracker = namedtuple('AB3DMOTTracker', ['info', 'tracker'])
# MotionModel = namedtuple('MotionModel', ['model', 'fx', 'hx', 'residual_fn', 'mode_prob', 'transition_prob'])

# AB3DMOT
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys
from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

NUSCENES_TRACKING_NAMES = [
    # 'bicycle',
    'bus',
    'car',
    # 'motorcycle',
    # 'pedestrian',
    'truck'
    # 'trailer'
]

def format_sample_result(sample_token, tracking_name, tracker):
    '''
    Input:
      tracker: (9): [h, w, l, x, y, z, rot_y], tracking_id, tracking_score
    Output:
    sample_result {
      "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
      "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
      "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
      "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
      "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
      "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
      "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                         Note that the tracking_name cannot change throughout a track.
      "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                         We average over frame level scores to compute the track level score.
                                         The score is used to determine positive and negative tracks via thresholding.
    }
    '''
    rotation = Quaternion(axis=[0, 0, 1], angle=tracker[2]).elements
    sample_result = {
        'sample_token': sample_token,
        'translation': [tracker[0], tracker[1], tracker[3]],
        'size': [tracker[4], tracker[5], tracker[6]],
        'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
        'velocity': [0, 0],
        'tracking_id': str(int(tracker[7])),
        'tracking_name': tracking_name,
        'tracking_score': tracker[8]
    }

    return sample_result

def greedy_match(distance_matrix, max_dist = 1000000):
    '''
    Find the one-to-one matching using greedy allgorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    '''
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks


    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[
            detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            if distance_matrix[detection_id, tracking_id] >= max_dist:
                break
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices


def associate_iou(dets=None, trks=None, dets_corners=None, trks_corners=None, yaw_pos=2, iou_threshold=0.1):

    reverse_matrix = np.zeros((len(dets), len(trks)), dtype=np.bool)
    distance_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), reverse_matrix, np.arange(len(dets)), np.empty((0, 8, 3), dtype=int)
    for d, (det, det_corners) in enumerate(zip(dets, dets_corners)):
        for t, (trk, trk_corners) in enumerate(zip(trks, trks_corners)):
            # distance of center is larger than two maximum length
            if not bbox_adjacency_2d(det, trk):
                continue
            distance_matrix[d, t] = -bbox_iou2d(det_corners, trk_corners)  # det: 8 x 3, trk: 8 x 3
            diff = np.expand_dims(det - trk, axis=1)  # 7 x 1
            # manual reversed angle
            diff[yaw_pos] = angle_in_range(dets[d][yaw_pos] - trks[t][yaw_pos])
            if abs(diff[yaw_pos]) > np.pi / 2.0:
                diff[yaw_pos] += np.pi
                diff[yaw_pos] = angle_in_range(diff[yaw_pos])
                reverse_matrix[d, t] = True

    if match_algorithm == 'greedy':
        # to_max_mask = distance_matrix > -iou_threshold
        # distance_matrix[to_max_mask] = 0
        matched_indices = greedy_match(distance_matrix)
    elif match_algorithm == 'hungarian':
        to_max_mask = distance_matrix > -iou_threshold
        distance_matrix[to_max_mask] = 0
        matched_indices = linear_assignment(distance_matrix)  # hungarian algorithm

    unmatched_detections = []
    for d in range(len(dets)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trks)):
        if len(matched_indices) == 0 or (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        match = True
        if distance_matrix[m[0], m[1]] > -iou_threshold:
            match = False
        if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, reverse_matrix, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate(dets=None, trks=None, trks_S=None, yaw_pos=2, mahalanobis_threshold=0.1, print_debug=False, match_algorithm='greedy'):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    dets: N x 7
    trks: M x 7
    trks_S: N x 7 x 7
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    reverse_matrix = np.zeros((len(dets), len(trks)), dtype=np.bool)
    distance_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for t, (trk, trk_S) in enumerate(zip(trks, trks_S)):
        S_inv = np.linalg.inv(trk_S)  # 7 x 7
        for d, det in enumerate(dets):
            diff = np.expand_dims(det - trk, axis=1)  # 7 x 1
            # manual reversed angle by 180 when diff > 90 or < -90 degree
            if yaw_pos > -1:
                diff[yaw_pos] = angle_in_range(diff[yaw_pos])
                if abs(diff[yaw_pos]) > np.pi / 2.0: # 180 deg diff
                    diff[yaw_pos] += np.pi
                    diff[yaw_pos] = angle_in_range(diff[yaw_pos])
                    reverse_matrix[d, t] = True
            distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
    if match_algorithm == 'greedy':
        # to_max_mask = distance_matrix > mahalanobis_threshold
        # distance_matrix[to_max_mask] = mahalanobis_threshold + 1
        matched_indices = greedy_match(distance_matrix)
    elif match_algorithm == 'hungarian':
        to_max_mask = distance_matrix > mahalanobis_threshold
        distance_matrix[to_max_mask] = mahalanobis_threshold + 1
        matched_indices = linear_assignment(distance_matrix)  # hungarian algorithm

    unmatched_detections = []
    for d in range(len(dets)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trks)):
        if len(matched_indices) == 0 or (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    matched_detections = []
    matched_trackers = []
    for m in matched_indices:
        match = True
        if distance_matrix[m[0], m[1]] > mahalanobis_threshold:
            # match = False
        # if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            matched_detections.append(m[0])
            matched_trackers.append(m[1])

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, reverse_matrix, np.array(matched_detections), np.array(matched_trackers), np.array(unmatched_detections), np.array(unmatched_trackers)

def samac_associate(m_dets=None, m_trks=None, m_trks_S=None, a_dets=None, a_trks=None, a_trks_S=None, trackers=None, m_state_on=None, a_state_on=None, da_state_on=None, m_yaw_pos=2, a_yaw_pos=-1, mdist_thres=None):
    reverse_matrix = np.zeros((len(m_dets), len(m_trks)), dtype=np.bool)
    m_likelihood = np.full((len(m_dets), len(m_trks)), 1.0, dtype=np.float32)
    a_likelihood = np.full((len(m_dets), len(m_trks)), 1.0, dtype=np.float32)
    m_prior = np.zeros((len(m_trks)), dtype=np.float32)
    a_prior = np.zeros((len(m_trks)), dtype=np.float32)
    max_dist = 50.0
    min_dist = 0.0



    a_da_state_on = da_state_on[a_state_on].copy()
    a_dets = a_dets[:, a_da_state_on, :]
    a_trks = a_trks[:, a_da_state_on, :]
    dim_a_da = np.count_nonzero(a_da_state_on)
    a_da_trks_S = np.zeros((len(a_trks), dim_a_da, dim_a_da))
    a_da_trks_S[:, np.ones((dim_a_da, dim_a_da), dtype=bool)] = a_trks_S[:, a_da_state_on.reshape(-1, 1).dot(a_da_state_on.reshape(-1, 1).T)]
    for t, m_trk in enumerate(m_trks):
        # if trackers[t].info.wscore < 0.3:
        #     a_state_on[1] = False
        # elif trackers[t].info.wscore > 0.7:
        #     a_state_on[2] = False
        m_S_inv = np.linalg.inv(m_trks_S[t])
        a_S_inv = np.linalg.inv(a_da_trks_S[t])
        d_count = 0
        for d, m_det in enumerate(m_dets):
            m_diff = (m_det - m_trk).reshape((-1, 1))
            # manual reversed angle by 180 when diff > 90 or < -90 degree
            if -1 < m_yaw_pos < len(m_diff):
                m_diff[m_yaw_pos] = angle_in_range(m_diff[m_yaw_pos])
                if abs(m_diff[m_yaw_pos]) > np.pi / 2.0: # 180 deg diff
                    m_diff[m_yaw_pos] += np.pi
                    m_diff[m_yaw_pos] = angle_in_range(m_diff[m_yaw_pos])
                    reverse_matrix[d, t] = True
            m_mdist = np.sqrt(np.dot(np.dot(m_diff.T, m_S_inv), m_diff))
            if m_mdist < mdist_thres['motion'][len(m_diff)]:
                d_count += 1
                a_diff = (a_dets[d, :] - a_trks[t, :]).reshape((-1, 1))
                if -1 < a_yaw_pos < len(a_diff):
                    a_diff[a_yaw_pos] = angle_in_range(a_diff[a_yaw_pos])
                    if abs(a_diff[a_yaw_pos]) > np.pi / 2.0:  # 180 deg diff
                        a_diff[a_yaw_pos] += np.pi
                        a_diff[a_yaw_pos] = angle_in_range(a_diff[a_yaw_pos])
                        reverse_matrix[d, t] = True
                a_mdist = np.sqrt(np.dot(np.dot(a_diff.T, a_S_inv), a_diff))
                m_likelihood[d, t] = -(mdist_thres['motion'][len(m_diff)] - m_mdist)/mdist_thres['motion'][len(m_diff)]
                if a_mdist < mdist_thres['appearance'][len(a_diff)]:
                    a_likelihood[d, t] = -(mdist_thres['appearance'][len(a_diff)] - a_mdist)/mdist_thres['appearance'][len(a_diff)]
                else:
                    a_likelihood[d, t] = 0.0
                    # m_likelihood = -smn.pdf(det[0:3][m_state_on], trk[0:3][m_state_on], m_trk_S) # -np.exp(logpdf(det[0:3][m_state_on], trk[0:3][m_state_on], m_trk_S)) # -smn(trk[0:3][m_state_on], m_trk_S).pdf(det[0:3][m_state_on])
                # m_flike = fmn(det[0:3][m_state_on], trk[0:3][m_state_on], m_trk_S)
                # a_likelihood = -smn.pdf(det[3:7][a_state_on], trk[3:7][a_state_on], a_trk_S) # -np.exp(logpdf(det[3:7][a_state_on], trk[3:7][a_state_on], a_trk_S)) # -smn(trk[3:7][a_state_on], a_trk_S).pdf(det[3:7][a_state_on]))
                # a_flike = fmn(det[3:7][a_state_on], trk[3:7][a_state_on], a_trk_S)
        if d_count > 0:
            close_score = (max_dist - min((max_dist - min_dist), max(trackers[t].info.dist - min_dist, 0.0))) / max_dist
            crowd_score = 0.5 * (1.0 + min(3.0, d_count - 1)/3.0)
            rigid_score = 0.5 * (1.0 + len(a_diff)/5.0)
            a_prior[t] = 0.5 * close_score * crowd_score * rigid_score
            m_prior[t] = 1.0 - a_prior[t]
        likelihood = m_prior*m_likelihood + a_prior*a_likelihood
    match_algorithm = 'greedy'
    if match_algorithm == 'greedy':
        matched_indices = greedy_match(likelihood, 0.0)
    elif match_algorithm == 'hungarian':
        matched_indices = linear_assignment(likelihood)
        remove_indices = []
        for i, (midx, tidx) in enumerate(matched_indices):
            if likelihood[midx, tidx] > 0.0:
                remove_indices.append(i)
            matched_indices = np.delete(matched_indices, remove_indices, axis=0)
    else:
        pass

    unmatched_trks = np.arange(len(m_trks))
    unmatched_dets = np.arange(len(m_dets))
    if len(matched_indices) > 0:
        matched_dets = matched_indices[:, 0]
        matched_trks = matched_indices[:, 1]
        unmatched_dets = np.delete(unmatched_dets, matched_dets)
        unmatched_trks = np.delete(unmatched_trks, matched_trks)

    return matched_indices, reverse_matrix, unmatched_dets, unmatched_trks


class AB3DMOT(object):
    def __init__(self, covariance_id=0, max_tracking_age=3, max_new_age=2, min_hits=2, tracking_name='car', use_angular_velocity=False,
                 tracking_nuscenes=False):

        self.max_tracking_age = max_tracking_age
        self.max_new_age = max_new_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        """
        observation:
          before reorder: [x, y, z, yaw, w, l, h]
          after reorder:  [x, y, yaw, z, w, l, h]
        state:
          [x, y, yaw, z, w, l, h]
        """
        self.reorder_samac = [0, 1, 3, 2, 4, 5, 6]
        self.reorder_back_samac = [0, 1, 3, 2, 4, 5, 6]

        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_nuscenes = tracking_nuscenes
        self.tracker_generator = TrackerGenerator(self.tracking_name)
        cov = Covariance(1)
        self.mdist_thres = cov.mdist_threshold

    def samac_update(self, dets_all, match_distance, match_algorithm, pose_translation, pose_rotation, cs_translation, cs_rotation):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # print(self.tracking_name, "==========================")
        yaw_pos = 2
        print_debug = False
        self.frame_count += 1
        dets, info, pts = dets_all['dets'], dets_all['info'], dets_all['pts']  # dets: N x 7, float numpy array
        # trks = []
        # trks_S = []
        ret = []
        # trks_tracking = []
        # trks_info = []
        m_trks = []
        a_trks = []
        m_trks_S = []
        a_trks_S = []
        # PREDICTION
        for t, trk in enumerate(self.trackers):
            trk.info.predict()
            m_pred_z, m_S = trk.mtracker.predict()
            a_pred_z, a_S = trk.atracker.predict()
            # print(m_pred_z.shape, a_pred_z.shape, m_S.shape, a_S.shape)
            m_trks.append(m_pred_z)
            a_trks.append(a_pred_z)
            m_trks_S.append(m_S)
            a_trks_S.append(a_S)
            pred_z = np.zeros((7, 1))
            pred_z[self.tracker_generator.m_state_on] = m_pred_z
            pred_z[self.tracker_generator.a_state_on] = a_pred_z
            # wscore, dist, detection_score######################### more info
            tbox = Box(center=[pred_z[0], pred_z[1], pred_z[3]], size=[pred_z[4], pred_z[5], pred_z[6]],
                       orientation=Quaternion(axis=[0., 0., 1.], angle=pred_z[2]))
            tbox.translate(pose_translation)
            tbox.rotate(pose_rotation)
            tbox.translate(cs_translation)
            tbox.rotate(cs_rotation)
            loc_pos = np.arctan2(tbox.center[1], tbox.center[0])
            loc_yaw = quaternion_yaw(tbox.orientation)
            wscore = 0.5 * (1.0 + np.cos(2.0 * (loc_yaw - loc_pos)))
            dist = np.sqrt(tbox.center[0] ** 2 + tbox.center[1] ** 2)
            trk.info.wscore = wscore
            trk.info.dist = dist
            ########################################################

        # DATA ASSOCIATION
        if len(m_trks) > 0 and len(dets) > 0:
            m_trks = np.stack(m_trks, axis=0)
            m_trks_S = np.stack(m_trks_S, axis=0)
            a_trks = np.stack(a_trks, axis=0)
            a_trks_S = np.stack(a_trks_S, axis=0)
            m_dets = dets[:, self.tracker_generator.m_state_on, np.newaxis].copy()
            a_dets = dets[:, self.tracker_generator.a_state_on, np.newaxis].copy()
            if match_distance == 'test':
                matched, reverse, unmatched_dets, unmatched_trks = samac_associate(m_dets=m_dets, m_trks=m_trks, m_trks_S=m_trks_S,
                                                                                   a_dets=a_dets, a_trks=a_trks, a_trks_S=a_trks_S,
                                                                                   trackers=self.trackers,
                                                                                   m_state_on=self.tracker_generator.m_state_on,
                                                                                   a_state_on=self.tracker_generator.a_state_on,
                                                                                   da_state_on=self.tracker_generator.da_state_on,
                                                                                   m_yaw_pos=self.tracker_generator.m_yaw_pos, a_yaw_pos=self.tracker_generator.a_yaw_pos,
                                                                                   mdist_thres=self.mdist_thres)
            # elif match_distance == 'mahal':
            #     comp_dets = dets[:, self.tracker_generator.state_on]
            #     comp_trks = trks[:, self.tracker_generator.state_on]
            #     comp_trks_S = np.zeros((len(trks), self.tracker_generator.dim_z, self.tracker_generator.dim_z))
            #     comp_trks_S[:, np.ones((self.tracker_generator.dim_z, self.tracker_generator.dim_z), dtype=bool)] = trks_S[:, self.tracker_generator.state_cov_on]
            #     matched, reverse, matched_dets, matched_trks, unmatched_dets, unmatched_trks = associate(dets=comp_dets, trks=comp_trks, trks_S=comp_trks_S, yaw_pos=self.tracker_generator.yaw_pos,
            #                                                              mahalanobis_threshold=self.mdist_thres,
            #                                                              print_debug=print_debug,
            #                                                              match_algorithm=match_algorithm)
            #     # relationship with matched and unmatched detection
            # elif match_distance == 'icp':
            #     matched, reverse, unmatched_dets, unmatched_trks = samac_associate(dets=dets, trks=trks, trks_S=trks_S, trackers=self.trackers, trks_info=trks_info, trks_tracking=trks_tracking, yaw_pos=self.tracker_generator.yaw_pos, mdist_thres=self.mdist_thres, iou_thres=0.1)


        else:
            matched, reverse, unmatched_dets, unmatched_trks = np.empty((0, 2), dtype=int), np.zeros((len(dets), len(m_trks)), dtype=np.bool), np.arange(len(dets)), np.arange(len(m_trks))

        for idx in matched:
            bbox3d = dets[idx[0], :]
            if reverse[idx[0], idx[1]]:
                bbox3d[yaw_pos] += np.pi
                bbox3d[yaw_pos] = angle_in_range(bbox3d[yaw_pos])
            self.trackers[idx[1]].info.update(info[idx[0], :][0])
            self.trackers[idx[1]].mtracker.update(bbox3d[self.tracker_generator.m_state_on])
            self.trackers[idx[1]].atracker.update(bbox3d[self.tracker_generator.a_state_on])


        # NEW TRACK GENERATION
        # matched detection overlap with unmatched detection or updated tracks? => delete unmatched detection # the certain ratio
        for i in unmatched_dets:
            trk = self.tracker_generator.generate_tracker(z=dets[i, :], track_score=info[i][0])
            self.trackers.append(trk)

        # TRACK MANAGEMENT
        i = len(self.trackers)
        # self.min_hits = 4
        for trk in reversed(self.trackers):
            i -= 1
            m_z = trk.mtracker.get_state()
            a_z = trk.atracker.get_state()
            bbox3d = np.concatenate((m_z, a_z))
            bbox3d = bbox3d[self.reorder_back_samac].reshape(1, -1).squeeze()
            if trk.info.tracking:
                if trk.info.time_since_update < self.max_tracking_age:
                    if trk.info.time_since_update == 0:
                        ret.append(np.concatenate((bbox3d, [trk.info.id + 1], [trk.info.track_score])).reshape(1, -1))
                else:
                    self.trackers.pop(i)
            else:
                # remove tracker right away when no two-times update # AP-wise correction
                if trk.info.time_since_update < self.max_new_age:
                    if trk.info.hits >= self.min_hits or trk.info.hit_streak > 0:
                        trk.info.tracking = True
                    if self.frame_count < self.min_hits or trk.info.tracking: # or trk.info.time_since_update == 0 : #or (trk.info.hits > 1 and trk.info.time_since_update == 0): # or trk.info.time_since_update == 0 : # (trk.info.hits > 1 and trk.info.time_since_update == 0):
                        ret.append(np.concatenate((bbox3d, [trk.info.id + 1], [trk.info.track_score])).reshape(1, -1))
                else:
                    self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)  # x, y, z, theta, w, l, h, ID, other info, confidence
        return np.empty((0, 15 + 7))




def track_nuscenes(data_split, covariance_id, match_distance, match_threshold, match_algorithm, save_root,
                   use_angular_velocity):
    '''
    submission {
      "meta": {
          "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
          "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
          "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
          "use_map":      <bool>  -- Whether this submission uses map data as an input.
          "use_external": <bool>  -- Whether this submission uses external data as an input.
      },
      "results": {
          sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
      }
    }

    '''
    save_dir = os.path.join(save_root, data_split);
    mkdir_if_missing(save_dir)
    scene_splits = None
    version = None
    # if 'train' == data_split:
    #     detection_file = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_train.json'
    #     data_root = '/media/marco/60348B1F348AF776/nuscene/raw/v1.0-trainval'
    #     version = 'v1.0-trainval'
    #     output_path = os.path.join(save_dir, 'results.json')
    #     scene_splits = splits.train
    # elif 'val' == data_split:
    #     detection_file = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_val.json'
    #     data_root = '/media/marco/60348B1F348AF776/nuscene/raw/v1.0-trainval'
    #     version = 'v1.0-trainval'
    #     output_path = os.path.join(save_dir, 'results.json')
    #     scene_splits = splits.val
    # elif 'test' == data_split:
    #     detection_file = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_test.json'
    #     data_root = '/media/marco/60348B1F348AF776/nuscene/raw/v1.0-test'
    #     version = 'v1.0-test'
    #     output_path = os.path.join(save_dir, 'results.json')
    #     scene_splits = splits.test
    # elif 'mini_val' == data_split:
    #     detection_file = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_mini_val.json'
    #     data_root = '/media/marco/60348B1F348AF776/nuscene/raw/v1.0-mini'
    #     version = 'v1.0-mini'
    #     output_path = os.path.join(save_dir, 'results.json')
    #     scene_splits = splits.mini_val
    # elif 'mini_train' == data_split:
    #     detection_file = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_mini_train.json'
    #     data_root = '/media/marco/60348B1F348AF776/nuscene/raw/v1.0-mini'
    #     version = 'v1.0-mini'
    #     output_path = os.path.join(save_dir, 'results.json')
    #     scene_splits = splits.mini_train
    if 'train' == data_split:
        detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_train.json'
        data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-trainval'
        version = 'v1.0-trainval'
        output_path = os.path.join(save_dir, 'results.json')
        scene_splits = splits.train
    elif 'val' == data_split:
        detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_val.json'
        data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-trainval'
        version = 'v1.0-trainval'
        output_path = os.path.join(save_dir, 'results.json')
        scene_splits = splits.val
    elif 'test' == data_split:
        detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_test.json'
        data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-test'
        version = 'v1.0-test'
        output_path = os.path.join(save_dir, 'results.json')
        scene_splits = splits.test
    elif 'mini_val' == data_split:
        detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_mini_val.json'
        data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
        output_path = os.path.join(save_dir, 'results.json')
        scene_splits = splits.mini_val
    elif 'mini_train' == data_split:
        detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_mini_train.json'
        data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
        output_path = os.path.join(save_dir, 'results.json')
        scene_splits = splits.mini_train
    else:
        print('No Dataset Split', data_split)
        assert(False)
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    point_on = False
    icp_on = False
    viz_on = False
    results = {}

    total_time = 0.0
    total_frames = 0

    with open(detection_file) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
    meta = data['meta']

    processed_scene_tokens = set()
    if viz_on:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)


    for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
        sample_token = all_results.sample_tokens[sample_token_idx]
        scene_token = nusc.get('sample', sample_token)['scene_token']
        if scene_token in processed_scene_tokens:
            continue
        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token

        # initialize tracker
        mot_trackers = {
        tracking_name: AB3DMOT(covariance_id, tracking_name=tracking_name, use_angular_velocity=use_angular_velocity,
                               tracking_nuscenes=True) for tracking_name in NUSCENES_TRACKING_NAMES}

        while current_sample_token != '':
            results[current_sample_token] = []
            dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            dinfos = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            dboxes = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            tboxes = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            pts = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            obj_pts = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

            lidar_sd_token = nusc.get('sample', current_sample_token)["data"]["LIDAR_TOP"]
            sd_record = nusc.get("sample_data", lidar_sd_token)
            cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
            if point_on:
                pts = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                # LIDAR
                lidar_data_path = nusc.get_sample_data_path(lidar_sd_token)
                # POSE
                sample_rec = nusc.get('sample', sd_record['sample_token'])

                # lidar_points = LidarPointCloud.from_file(lidar_data_path)

                lidar_points, times = LidarPointCloud.from_file_multisweep(nusc,
                                                                 sample_rec,
                                                                 sd_record['channel'],
                                                                 'LIDAR_TOP',
                                                                 nsweeps=5)


            for dbox in all_results.boxes[current_sample_token]:
                if dbox.detection_name not in NUSCENES_TRACKING_NAMES:
                    continue

                # x, y, z, yaw, w, l, h
                yaw = quaternion_yaw(Quaternion(dbox.rotation))
                yaw = angle_in_range(yaw)
                detection = np.array([dbox.translation[0], dbox.translation[1], yaw, dbox.translation[2], dbox.size[0], dbox.size[1], dbox.size[2]])
                box = Box(dbox.translation, dbox.size, Quaternion(dbox.rotation))
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)
                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)

                # more info
                dist = np.sqrt(box.center[0]**2 + box.center[1]**2)
                loc_pos = np.arctan2(box.center[1], box.center[0])
                loc_yaw = quaternion_yaw(box.orientation)
                wscore = 0.5*(1.0 + np.cos(2.0*(loc_yaw - loc_pos)))
                if icp_on:
                    mask = points_in_box(box, lidar_points.points[0:3, :], wlh_factor=1.0)
                    pts[dbox.detection_name].append(lidar_points.points[0:3, mask])
                ##########################################################

                dets[dbox.detection_name].append(detection) # detection state
                info[dbox.detection_name].append(np.array([dbox.detection_score, dist, wscore])) # detection score
                dboxes[dbox.detection_name].append(box)
                dinfos[dbox.detection_name].append(np.array([dist, loc_pos, loc_yaw, wscore]))


            # if point_on:
            #     dets_all = {
            #     tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name]), 'boxes': dboxes[tracking_name],
            #                     'pts': pts[tracking_name]} for tracking_name in NUSCENES_TRACKING_NAMES}
            # else:

            dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name]), 'pts': pts[tracking_name]}
                        for tracking_name in NUSCENES_TRACKING_NAMES}
            total_frames += 1
            start_time = time.time()
            for tracking_name in NUSCENES_TRACKING_NAMES:
                trackers = mot_trackers[tracking_name].samac_update(dets_all=dets_all[tracking_name], match_distance=match_distance, match_algorithm=match_algorithm,
                                                                    pose_translation=-np.array(pose_record["translation"]).reshape(-1, 1), pose_rotation=Quaternion(pose_record["rotation"]).inverse,
                                                                    cs_translation=-np.array(cs_record["translation"]).reshape(-1, 1), cs_rotation=Quaternion(cs_record["rotation"]).inverse)

                # x, y, z, theta, w, l, h
                for tracker in trackers:
                    sample_result = format_sample_result(current_sample_token, tracking_name, tracker)
                    results[current_sample_token].append(sample_result)
                    if viz_on:
                        # tbox = Box((trackers[i][0], trackers[i][1], trackers[i][2]), (trackers[i][4], trackers[i][5], trackers[i][6]), Quaternion(axis=[0., 0., 1.], angle=trackers[i][3]))
                        tbox = Box((tracker[0], tracker[1], tracker[2]), (tracker[4], tracker[5], tracker[6]), Quaternion(axis=[0., 0., 1.], angle=tracker[3]))
                        tboxes[tracking_name].append(tbox)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # visualization
            if viz_on:
                ax.plot(0, 0, '+', color='black')
                for box, info in zip(dboxes['truck'], info['truck']):
                    # box.translate(-np.array(pose_record['translation']))
                    # box.rotate(Quaternion(pose_record['rotation']).inverse)
                    # box.translate(-np.array(cs_record['translation']))
                    # box.rotate(Quaternion(cs_record['rotation']).inverse)
                    box.wlh = box.wlh * 1.2
                    box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=3)
                    ax.text(box.center[0], box.center[1], "D({:.2f}, {:.2f}, {:.2f})".format(info[0], info[1], info[2]), fontsize=10)

                # for ttracking_name in tboxes:
                for box in tboxes['car']:
                    box.translate(-np.array(pose_record['translation']))
                    box.rotate(Quaternion(pose_record['rotation']).inverse)
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)
                    # ax.text(box.center[0], box.center[1], "T({}, {})".format(box.center[0], box.center[1]), fontsize=10)

                # if point_on:
                #     for pt in pts['truck']:
                #         ax.scatter(pt[0, :], pt[1, :], s=1.0)
                #     # ax.scatter(lidar_points.points[0, :], lidar_points.points[1, :], s=0.1)
                eval_range = 70
                axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
                ax.set_xlim(-axes_limit, axes_limit)
                ax.set_ylim(-axes_limit, axes_limit)
                plt.title(scene_token + ': ' + current_sample_token)
                plt.pause(1e-3)
                ax.clear()


            # get next frame and continue the while loop
            current_sample_token = nusc.get('sample', current_sample_token)['next']

        # left while loop and mark this scene as processed
        processed_scene_tokens.add(scene_token)

    # finished tracking all scenes, write output data
    output_data = {'meta': meta, 'results': results}
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile)

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

def track_kitti(cov_id, assocition_metric, association_method, result_sha, save_root):
	det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    seq_file_list, num_seq = load_list_from_folder(os.path.join('data/KITTI', result_sha))
    total_time, total_frames = 0.0, 0
    save_dir = os.path.join(save_root, result_sha);
    mkdir_if_missing(save_dir)
    eval_dir = os.path.join(save_dir, 'data');
    mkdir_if_missing(eval_dir)
    seq_count = 0

    for seq_file in seq_file_list:
        _, seq_name, _ = fileparts(seq_file)
        eval_file = os.path.join(eval_dir, seq_name + '.txt');
        eval_file = open(eval_file, 'w')
        save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name);
        mkdir_if_missing(save_trk_dir)

        mot_tracker = AB3DMOT()
        seq_dets = np.loadtxt(seq_file, delimiter=',')  # load detections, N x 15

        # if no detection in a sequence
        if len(seq_dets.shape) == 1: seq_dets = np.expand_dims(seq_dets, axis=0)
        if seq_dets.shape[1] == 0:
            eval_file.close()
            continue

        # loop over frame
        min_frame, max_frame = int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max())
        for frame in range(min_frame, max_frame + 1):
            # logging
            print_str = 'processing %s: %d/%d, %d/%d   \r' % (seq_name, seq_count, num_seq, frame, max_frame)
            sys.stdout.write(print_str)
            sys.stdout.flush()
            save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame);
            save_trk_file = open(save_trk_file, 'w')

            # get irrelevant information associated with an object, not used for associationg
            ori_array = seq_dets[seq_dets[:, 0] == frame, -1].reshape((-1, 1))  # orientation
            other_array = seq_dets[seq_dets[:, 0] == frame, 1:7]  # other information, e.g, 2D box, ...
            additional_info = np.concatenate((ori_array, other_array), axis=1)

            dets = seq_dets[seq_dets[:, 0] == frame,
                   7:14]  # h, w, l, x, y, z, theta in camera coordinate follwing KITTI convention
            dets_all = {'dets': dets, 'info': additional_info}

            # important
            start_time = time.time()
            trackers = mot_tracker.update(dets_all)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # saving results, loop over each tracklet
            for d in trackers:
                bbox3d_tmp = d[0:7]  # h, w, l, x, y, z, theta in camera coordinate
                id_tmp = d[7]
                ori_tmp = d[8]
                type_tmp = det_id2str[d[9]]
                bbox2d_tmp_trk = d[10:14]
                conf_tmp = d[14]

                # save in detection format with track ID, can be used for dection evaluation and tracking visualization
                str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
                                                                                         bbox2d_tmp_trk[0],
                                                                                         bbox2d_tmp_trk[1],
                                                                                         bbox2d_tmp_trk[2],
                                                                                         bbox2d_tmp_trk[3],
                                                                                         bbox3d_tmp[0], bbox3d_tmp[1],
                                                                                         bbox3d_tmp[2], bbox3d_tmp[3],
                                                                                         bbox3d_tmp[4], bbox3d_tmp[5],
                                                                                         bbox3d_tmp[6], conf_tmp,
                                                                                         id_tmp)
                save_trk_file.write(str_to_srite)

                # save in tracking format, for 3D MOT evaluation
                str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp,
                                                                                          type_tmp, ori_tmp,
                                                                                          bbox2d_tmp_trk[0],
                                                                                          bbox2d_tmp_trk[1],
                                                                                          bbox2d_tmp_trk[2],
                                                                                          bbox2d_tmp_trk[3],
                                                                                          bbox3d_tmp[0], bbox3d_tmp[1],
                                                                                          bbox3d_tmp[2], bbox3d_tmp[3],
                                                                                          bbox3d_tmp[4], bbox3d_tmp[5],
                                                                                          bbox3d_tmp[6],
                                                                                          conf_tmp)
                eval_file.write(str_to_srite)

            total_frames += 1
            save_trk_file.close()
        seq_count += 1
        eval_file.close()
    print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))




if __name__ == '__main__':
    if len(sys.argv) != 9:
        print(
            "Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root")
        sys.exit(1)

    data_split = sys.argv[1]
    covariance_id = int(sys.argv[2])
    match_distance = sys.argv[3]
    match_algorithm = sys.argv[5]
    use_angular_velocity = sys.argv[6] == 'True' or sys.argv[6] == 'true'
    dataset = sys.argv[7]

    save_root = os.path.join('./' + sys.argv[8])

    # cov_id, association_score (mdist, dist, iou), association_method (greedy, hungarian, jpda)
    if dataset == 'kitti':
        print('track kitti not supported')
        track_kitti(cov_id, association_metric, association_method, result_sha, save_root)
    elif dataset == 'nuscenes':
        print('track nuscenes')
        track_nuscenes(cov_id, association_metric, association_method, data_split, save_root)
    # elif dataset == 'argoverse':
    #     print('track argoverse')


