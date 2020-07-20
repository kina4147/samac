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

# AB3DMOTTracker = namedtuple('AB3DMOTTracker', ['info', 'tracker'])
# MotionModel = namedtuple('MotionModel', ['model', 'fx', 'hx', 'residual_fn', 'mode_prob', 'transition_prob'])

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
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
            if distance_matrix[detection_id, tracking_id] > max_dist:
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

    # assert(False)

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

def samac_associate(dets=None, trks=None, trks_S=None, dets_corners=None, trks_info=None, trks_tracking=None, yaw_pos=2, mdist_thres=11, iou_thres=0.1):

    reverse_matrix = np.zeros((len(dets), len(trks)), dtype=np.bool)
    distance_matrix = np.full((len(dets), len(trks)), mdist_thres + 1.0, dtype=np.float32)
    for t, (trk, trk_S) in enumerate(zip(trks, trks_S)):
        S_inv = np.linalg.inv(trk_S)
        for d, det in enumerate(dets):
            # diff = np.expand_dims(det - trk, axis=1)  # 7 x 1
            diff = (det - trk).reshape((-1, 1))
            # manual reversed angle by 180 when diff > 90 or < -90 degree
            if yaw_pos > -1:
                diff[yaw_pos] = angle_in_range(diff[yaw_pos])
                if abs(diff[yaw_pos]) > np.pi / 2.0: # 180 deg diff
                    diff[yaw_pos] += np.pi
                    diff[yaw_pos] = angle_in_range(diff[yaw_pos])
                    reverse_matrix[d, t] = True
            distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])

    distance_matrix[distance_matrix > mdist_thres] = mdist_thres + 1.0
    match_algorithm = 'greedy'
    if match_algorithm == 'greedy':
        matched_indices = greedy_match(distance_matrix, mdist_thres)
    elif match_algorithm == 'hungarian':
        matched_indices = linear_assignment(distance_matrix)  # hungarian algorithm
        remove_indices = []
        for i, (midx, tidx) in enumerate(matched_indices):
            if distance_matrix[midx, tidx] > mdist_thres:
                remove_indices.append(i)
            matched_indices = np.delete(matched_indices, remove_indices, axis=0)
    else:
        pass

    unmatched_trks = np.arange(len(trks))
    unmatched_dets = np.arange(len(dets))
    if len(matched_indices) > 0:
        matched_dets = matched_indices[:, 0]
        matched_trks = matched_indices[:, 1]
        unmatched_dets = np.delete(unmatched_dets, matched_dets)
        unmatched_trks = np.delete(unmatched_trks, matched_trks)
        overlapped_unmatched_dets = []
        # for md in matched_dets:
        for midx, tidx in matched_indices:
            # if trks_tracking[mt]:
            for umidx in unmatched_dets:
                # if adjacency_2d(dets[midx], dets[umidx]):
                iou2d = bbox_iou2d(dets_corners[midx], dets_corners[umidx], True)
                if iou2d > 0.3:
                    overlapped_unmatched_dets.append(umidx)
        unmatched_dets = np.delete(unmatched_dets, overlapped_unmatched_dets)

    return matched_indices, reverse_matrix, unmatched_dets, unmatched_trks

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
    # if len(trks) == 0:
    #     return np.empty((0, 2), dtype=int), reverse_matrix, np.arange(len(dets)), np.empty((0, 8, 3), dtype=int)
    # assert (dets is not None)
    # assert (trks is not None)
    # assert (trks_S is not None)
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

class AB3DMOT(object):
    def __init__(self, covariance_id=0, max_tracking_age=3, max_new_age=1, min_hits=3, tracking_name='car', use_angular_velocity=False,
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

        """
        observation:
          before reorder: [h, w, l, x, y, z, rot_y]
          after reorder:  [x, y, z, rot_y, l, w, h]
        state:
          [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
        """
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]

        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_nuscenes = tracking_nuscenes
        self.tracker_generator = TrackerGenerator(self.tracking_name)

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
        yaw_pos = 2
        print_debug = False
        self.frame_count += 1
        dets, info = dets_all['dets'], dets_all['info']  # dets: N x 7, float numpy array
        trks = []
        trks_S = []
        ret = []
        trks_tracking = []
        trks_info = []
        # PREDICTION
        for t, trk in enumerate(self.trackers):
            trk.info.predict()
            a_pred_z, a_S = trk.atracker.predict()
            m_pred_z, m_S = trk.mtracker.predict()
            pred_z = np.concatenate((m_pred_z, a_pred_z)) # x, y, rot_y, z, h, w, l
            S = block_diag(m_S, a_S)

            # wscore, dist, detection_score######################### more info
            tbox = Box(center=[pred_z[0], pred_z[1], pred_z[3]], size=[pred_z[4], pred_z[5], pred_z[6]], orientation=Quaternion(axis=[0., 0., 1.], angle=pred_z[2]))
            tbox.translate(pose_translation)
            tbox.rotate(pose_rotation)
            tbox.translate(cs_translation)
            tbox.rotate(cs_rotation)
            loc_pos = np.arctan2(tbox.center[1], tbox.center[0])
            loc_yaw = quaternion_yaw(tbox.orientation)
            wscore = 0.5*(1.0 + np.cos(2.0*(loc_yaw - loc_pos)))
            dist = np.sqrt(tbox.center[0]**2 + tbox.center[1]**2)
            #######################################
            trks.append(pred_z.squeeze())
            trks_S.append(S)
            trks_tracking.append(trk.info.tracking)
            trks_info.append(np.array([dist, wscore]))

        # association
        if len(trks) > 0 and len(dets) > 0:
            trks = np.stack(trks, axis=0)
            trks_S = np.stack(trks_S, axis=0)
            # DATA ASSOCIATION
            if match_distance == 'test':
                dets_corners = [corners(det) for det in dets]
                comp_dets = dets[:, self.tracker_generator.state_on]
                comp_trks = trks[:, self.tracker_generator.state_on]
                comp_trks_S = np.zeros((len(trks), self.tracker_generator.dim_z, self.tracker_generator.dim_z))
                comp_trks_S[:, np.ones((self.tracker_generator.dim_z, self.tracker_generator.dim_z), dtype=bool)] = trks_S[:, self.tracker_generator.state_cov_on]
                matched, reverse, unmatched_dets, unmatched_trks = samac_associate(dets=dets, trks=trks, trks_S=trks_S, dets_corners=dets_corners, trks_info=trks_info, trks_tracking= trks_tracking, yaw_pos=self.tracker_generator.yaw_pos, mdist_thres=self.tracker_generator.mdist_thres, iou_thres=0.1)
            elif match_distance == 'mahal':
                comp_dets = dets[:, self.tracker_generator.state_on]
                comp_trks = trks[:, self.tracker_generator.state_on]
                comp_trks_S = np.zeros((len(trks), self.tracker_generator.dim_z, self.tracker_generator.dim_z))
                comp_trks_S[:, np.ones((self.tracker_generator.dim_z, self.tracker_generator.dim_z), dtype=bool)] = trks_S[:, self.tracker_generator.state_cov_on]
                matched, reverse, matched_dets, matched_trks, unmatched_dets, unmatched_trks = associate(dets=comp_dets, trks=comp_trks, trks_S=comp_trks_S, yaw_pos=self.tracker_generator.yaw_pos,
                                                                         mahalanobis_threshold=self.tracker_generator.mdist_thres,
                                                                         print_debug=print_debug,
                                                                         match_algorithm=match_algorithm)
                # relationship with matched and unmatched detection

        else:
            matched, reverse, unmatched_dets, unmatched_trks = np.empty((0, 2), dtype=int), np.zeros((len(dets), len(trks)), dtype=np.bool), np.arange(len(dets)), np.arange(len(trks))

        for idx in matched:
            bbox3d = dets[idx[0], :]
            if reverse[idx[0], idx[1]]:
                bbox3d[yaw_pos] += np.pi
                bbox3d[yaw_pos] = angle_in_range(bbox3d[yaw_pos])
            self.trackers[idx[1]].info.update(info[idx[0], :], info[idx[0], :][0])
            self.trackers[idx[1]].mtracker.update(bbox3d[0:3])
            self.trackers[idx[1]].atracker.update(bbox3d[3:7])


        # NEW TRACK GENERATION
        # matched detection overlap with unmatched detection or updated tracks? => delete unmatched detection # the certain ratio
        for i in unmatched_dets:
            trk = self.tracker_generator.generate_tracker(z=dets[i, :], info=info[i], track_score=info[i][0])
            self.trackers.append(trk)

        # TRACK MANAGEMENT
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            m_z = trk.mtracker.get_state()
            a_z = trk.atracker.get_state()
            bbox3d = np.concatenate((m_z, a_z))
            bbox3d = bbox3d[self.reorder_back_samac].reshape(1, -1).squeeze()
            if trk.info.tracking:
                if trk.info.time_since_update < self.max_tracking_age: # and trk.info.hit_streak > 0:
                    ret.append(np.concatenate((bbox3d, [trk.info.id + 1], [trk.info.track_score])).reshape(1,
                                                                                                           -1))  # +1 as MOT benchmark requires positive
                else:
                    self.trackers.pop(i)
            else:
                if trk.info.time_since_update < self.max_new_age:
                    if trk.info.hits >= self.min_hits:
                        trk.info.tracking = True
                    if trk.info.hit_streak > 0 or self.frame_count <= self.min_hits:
                        ret.append(np.concatenate((bbox3d, [trk.info.id + 1], [trk.info.track_score])).reshape(1,
                                                                                                           -1))  # +1 as MOT benchmark requires positive

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

                lidar_points = LidarPointCloud.from_file(lidar_data_path)

                # lidar_points, times = LidarPointCloud.from_file_multisweep(nusc,
                #                                                  sample_rec,
                #                                                  sd_record['channel'],
                #                                                  'LIDAR_TOP',
                #                                                  nsweeps=5)


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
                # if dbox.detection_name == 'car':
                #     print(dist, box.center, loc_pos, loc_yaw, wscore)
                # point in box ###########################################
                # if point_on:
                #     # Create Box instance.
                #     box.translate(-np.array(pose_record["translation"]))
                #     box.rotate(Quaternion(pose_record["rotation"]).inverse)
                #     box.translate(-np.array(cs_record["translation"]))
                #     box.rotate(Quaternion(cs_record["rotation"]).inverse)
                #     mask = points_in_box(box, lidar_points.points[0:3, :], wlh_factor=1.0)
                #     pts[dbox.detection_name].append(lidar_points.points[0:3, mask])
                ##########################################################

                # information = np.array([dbox.detection_score])
                dets[dbox.detection_name].append(detection) # detection state
                info[dbox.detection_name].append(np.array([dbox.detection_score, dist, wscore])) # detection score
                dboxes[dbox.detection_name].append(box)
                dinfos[dbox.detection_name].append(np.array([dist, loc_pos, loc_yaw, wscore]))


            # if point_on:
            #     dets_all = {
            #     tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name]), 'boxes': dboxes[tracking_name],
            #                     'pts': pts[tracking_name]} for tracking_name in NUSCENES_TRACKING_NAMES}
            # else:

            dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])}
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
                for box, info in zip(dboxes['car'], info['car']):
                    # box.translate(-np.array(pose_record['translation']))
                    # box.rotate(Quaternion(pose_record['rotation']).inverse)
                    # box.translate(-np.array(cs_record['translation']))
                    # box.rotate(Quaternion(cs_record['rotation']).inverse)
                    box.wlh = box.wlh * 1.2
                    # box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=3)
                    # ax.text(box.center[0], box.center[1], "D({}, {:.2f}, {:.2f}, {:.2f})".format(int(info[0]), info[1], info[2], info[3]), fontsize=10)

                # for ttracking_name in tboxes:
                for box in tboxes['car']:
                    box.translate(-np.array(pose_record['translation']))
                    box.rotate(Quaternion(pose_record['rotation']).inverse)
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)
                    # ax.text(box.center[0], box.center[1], "T({}, {})".format(box.center[0], box.center[1]), fontsize=10)

                if point_on:
                    ax.scatter(lidar_points.points[0, :], lidar_points.points[1, :], s=0.1)
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

if __name__ == '__main__':
    if len(sys.argv) != 9:
        print(
            "Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root")
        sys.exit(1)

    data_split = sys.argv[1]
    covariance_id = int(sys.argv[2])
    match_distance = sys.argv[3]
    match_threshold = float(sys.argv[4])
    match_algorithm = sys.argv[5]
    use_angular_velocity = sys.argv[6] == 'True' or sys.argv[6] == 'true'
    dataset = sys.argv[7]

    save_root = os.path.join('./' + sys.argv[8])

    if dataset == 'kitti':
        print('track kitti not supported')
    elif dataset == 'nuscenes':
        print('track nuscenes')
        track_nuscenes(data_split, covariance_id, match_distance, match_threshold, match_algorithm, save_root,
                       use_angular_velocity)
    # elif dataset == 'argoverse':
    #     print('track argoverse')


