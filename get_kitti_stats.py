import os
import sys

import numpy as np
# from utils import iou3d, convert_3dbox_to_8corner
from sklearn.utils.linear_assignment_ import linear_assignment

# from nuscenes import NuScenes
# from nuscenes.eval.common.config import config_factory
# from nuscenes.eval.tracking.evaluate import TrackingEval
# from nuscenes.eval.detection.data_classes import DetectionConfig
# from nuscenes.eval.detection.data_classes import DetectionBox
# from nuscenes.eval.tracking.data_classes import TrackingBox
# from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
# from nuscenes.eval.tracking.loaders import create_tracks
from pyquaternion import Quaternion
# from nuscenes.eval.common.utils import quaternion_yaw
# from SAMAC3DMOT.utils import rotz, angle_in_range, bbox_iou2d, bbox_iou3d, bbox_adjacency_2d


import os.path, copy, numpy as np, time, sys
from pykitti.utils import load_oxts_packets_and_poses

from SAMAC3DMOT.utils import angle_in_range, quaternion_to_euler
import argparse

KITTI_CLASS_NAMES = [
  'Car',
  'Pedestrian',
  'Cyclist'
]

# def corners(bbox3d):
#     """
#     Draw 3d bounding box in image
#         qs: (8,3) array of vertices for the 3d box in following order:
#             5 -------- 4
#            /|         /|
#           6 -------- 7 .
#           | |        | |
#           . 1 -------- 0
#           |/         |/       z|__y
#           2 -------- 3         /x
#
#     Returns the bounding box corners.
#
#     :param wlh_factor: Multiply w, l, h by a factor to scale the box.
#     :return: <np.float: 3, 8>. First four corners are the ones facing forward.
#         The last four are the ones facing backwards.
#     """
#     # w, l, h =  * wlh_factor
#     w = bbox3d[4]
#     l = bbox3d[5]
#     h = bbox3d[6]
#
#     R = rotz(bbox3d[3])
#     # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
#     # x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
#     # y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
#     # z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
#     x_corners = l / 2 * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
#     y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
#     z_corners = h / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
#     corners = np.vstack((x_corners, y_corners, z_corners))
#
#     # Rotate
#     corners = np.dot(R, corners)
#
#     # Translate
#     corners[0, :] = corners[0, :] + bbox3d[0]
#     corners[1, :] = corners[1, :] + bbox3d[1]
#     corners[2, :] = corners[2, :] + bbox3d[2]
#
#     return corners
# def rotation_to_positive_z_angle(rotation):
#   q = Quaternion(rotation)
#   angle = q.angle if q.axis[2] > 0 else -q.angle
#   return angle
#
# def get_mean(tracks):
#   '''
#   Input:
#     tracks: {scene_token:  {t: [TrackingBox]}}
#   '''
#   print('len(tracks.keys()): ', len(tracks.keys()))
#   yaw_pos = 3
#   # x, y, yaw, z, w, l, h, x', y', yaw', z', w', l', h'
#   gt_trajectory_map = {tracking_name: {scene_token: {} for scene_token in tracks.keys()} for tracking_name in NUSCENES_TRACKING_NAMES}
#
#   # store every detection data to compute mean and variance
#   gt_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
#   # no_head_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # = np.array([0, 0, 0, 0])  # x, y, x', y'
#   # head_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # = np.array([0, 0, 0, 0, 0])  # x, y', yaw, v, yaw'
#   # appearance_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # = np.array([0, 0, 0, 0, 0])  # , z, w, l, h, z'
#
#   for scene_token in tracks.keys():
#     # print('scene_token: ', scene_token)
#     # print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
#     for t_idx in range(len(tracks[scene_token].keys())):
#       #print('t_idx: ', t_idx)
#       t = sorted(tracks[scene_token].keys())[t_idx]
#       # print(tracks[scene_token][t])
#       for box_id, box in enumerate(tracks[scene_token][t]): # range(len(tracks[scene_token][t])):
#         #print('box_id: ', box_id)
#         # box = tracks[scene_token][t][box_id]
#         #print('box: ', box)
#
#         if box.tracking_name not in NUSCENES_TRACKING_NAMES:
#           continue
#         # x, y, z, yaw, x', y', z', yaw', v, x", y", z", yaw", a
#         yaw = quaternion_yaw(Quaternion(box.rotation))
#         box_data = np.array([box.translation[0], box.translation[1], box.translation[2], yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#         box_data[yaw_pos] = angle_in_range(box_data[yaw_pos])
#         if box.tracking_id not in gt_trajectory_map[box.tracking_name][scene_token]:
#           gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id] = {t_idx: box_data}
#         else:
#           gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
#
#         # if we can find the same object in the previous frame, get the velocity
#         if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx-1 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
#           prev_box_data = gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1]
#           dist = np.sqrt((box_data[0] - prev_box_data[0])**2 + (box_data[1] - prev_box_data[1])**2)
#           residual_vel = box_data[:4] - prev_box_data[:4] # x, y, z, yaw
#           # angle in range
#           residual_vel[yaw_pos] = angle_in_range(residual_vel[yaw_pos])
#           box_data[4:8] = residual_vel
#           box_data[8] = dist
#           gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
#           # if we can find the same object in the previous two frames, get the acceleration
#           if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx-2 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
#             pprev_box_data = gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 2]
#             pdist = np.sqrt((prev_box_data[0] - pprev_box_data[0])**2 + (prev_box_data[1] - pprev_box_data[1])**2)
#             presidual_vel = prev_box_data[:4] - pprev_box_data[:4] # x, y, yaw
#             presidual_vel[yaw_pos] = angle_in_range(presidual_vel[yaw_pos])
#             residual_a = residual_vel - presidual_vel
#             residual_a[yaw_pos] = angle_in_range(residual_a[yaw_pos])
#             dist_a = dist - pdist
#             box_data[9:13] = residual_a
#             box_data[13] = dist_a
#             gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
#             gt_box_data[box.tracking_name].append(box_data)
#             # # x, y, z, yaw, x', y', z', yaw', v, x", y", z", yaw", a
#             # no_head_box_data[box.tracking_name].append(np.array([box_data[0], box_data[1], box_data[3], box_data[4], box_data[5], box_data[7]]))
#             # head_box_data[box.tracking_name].append(np.array([box_data[0], box_data[1], box_data[3], box_data[8], box_data[7]]))
#             # appearance_box_data[box.tracking_name].append(np.array([box_data[2], 0.0, 0.0, 0.0, box_data[6]]))
#
#   mean = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
#   std = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
#   var = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
#   for tracking_name in NUSCENES_TRACKING_NAMES:
#     if len(gt_box_data[tracking_name]):
#       gt_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)
#       # no_head_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)
#       # head_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)
#       # appearance_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)
#
#       mean[tracking_name] = np.mean(gt_box_data[tracking_name], axis=0)
#       std[tracking_name] = np.std(gt_box_data[tracking_name], axis=0)
#       var[tracking_name] = np.var(gt_box_data[tracking_name], axis=0)
#   #
#   # # x, y, z, yaw, x', y', z', yaw', v, x", y", z", yaw", a
#   # mean = {tracking_name: np.mean(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
#   # std = {tracking_name: np.std(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
#   # var = {tracking_name: np.var(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
#
#
#   # no_head_mean = {tracking_name: np.mean(no_head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   # no_head_std = {tracking_name: np.std(no_head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   # no_head_var = {tracking_name: np.var(no_head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   #
#   #
#   # head_mean = {tracking_name: np.mean(head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   # head_std = {tracking_name: np.std(head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   # head_var = {tracking_name: np.var(head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   #
#   #
#   # appearance_mean = {tracking_name: np.mean(appearance_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   # appearance_std = {tracking_name: np.std(appearance_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   # appearance_var = {tracking_name: np.var(appearance_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
#   return mean, std, var #, head_std, head_var, appearance_mean, appearance_std, appearance_var
#
#
# def greedy_match(distance_matrix, max_dist = 1000000):
#     '''
#     Find the one-to-one matching using greedy allgorithm choosing small distance
#     distance_matrix: (num_detections, num_tracks)
#     '''
#     matched_indices = []
#
#     num_detections, num_tracks = distance_matrix.shape
#     distance_1d = distance_matrix.reshape(-1)
#     index_1d = np.argsort(distance_1d)
#     index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
#     detection_id_matches_to_tracking_id = [-1] * num_detections
#     tracking_id_matches_to_detection_id = [-1] * num_tracks
#
#
#     for sort_i in range(index_2d.shape[0]):
#         detection_id = int(index_2d[sort_i][0])
#         tracking_id = int(index_2d[sort_i][1])
#         if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[
#             detection_id] == -1:
#             tracking_id_matches_to_detection_id[tracking_id] = detection_id
#             detection_id_matches_to_tracking_id[detection_id] = tracking_id
#             if distance_matrix[detection_id, tracking_id] > max_dist:
#                 break
#             matched_indices.append([detection_id, tracking_id])
#
#     matched_indices = np.array(matched_indices)
#     return matched_indices
#
# def matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks_gt, matching_dist):
#   '''
#   For each sample token, find matches of pred_boxes and gt_boxes, then get stats.
#   tracks_gt has the temporal order info for each sample_token
#   '''
#   yaw_pos = 3
#   diff = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
#   diff_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw']
#
#   # # similar to main.py class AB3DMOT update()
#   # reorder = [3, 4, 5, 6, 2, 1, 0]
#   # reorder_back = [6, 5, 4, 0, 1, 2, 3]
#
#   for scene_token in tracks_gt.keys():
#     #print('scene_token: ', scene_token)
#     #print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
#     # {tracking_name: t_idx: tracking_id: det(7) }
#     match_diff_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
#     match_gt_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
#     match_det_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
#     for t_idx in range(len(tracks_gt[scene_token].keys())):
#       #print('t_idx: ', t_idx)
#       t = sorted(tracks_gt[scene_token].keys())[t_idx]
#       #print(len(tracks_gt[scene_token][t]))
#       if len(tracks_gt[scene_token][t]) == 0:
#         continue
#       ref_box = tracks_gt[scene_token][t][0]
#       sample_token = ref_box.sample_token
#
#       for tracking_name in NUSCENES_TRACKING_NAMES:
#         #print('t: ', t)
#         gt_all = [box for box in gt_boxes.boxes[sample_token] if box.tracking_name == tracking_name]
#         if len(gt_all) == 0:
#           continue
#
#         gts = np.stack([np.array([box.translation[0], box.translation[1], box.translation[2], angle_in_range(quaternion_yaw(Quaternion(box.rotation))), box.size[0], box.size[1], box.size[2]]) for box in gt_all], axis=0)
#
#
#         gts_ids = [box.tracking_id for box in gt_all]
#         # gts = np.stack([np.array([
#         #   box.size[2], box.size[0], box.size[1],
#         #   box.translation[0], box.translation[1], box.translation[2],
#         #   rotation_to_positive_z_angle(box.rotation)
#         #   ]) for box in gt_all], axis=0)
#         # gts_ids = [box.tracking_id for box in gt_all]
#
#         det_all = [box for box in pred_boxes.boxes[sample_token] if box.detection_name == tracking_name]
#         if len(det_all) == 0:
#           continue
#
#         dets = np.stack([np.array([box.translation[0], box.translation[1], box.translation[2], angle_in_range(quaternion_yaw(Quaternion(box.rotation))), box.size[0], box.size[1], box.size[2]]) for box in det_all], axis=0)
#
#         #
#         # dets = dets[:, reorder]
#         # gts = gts[:, reorder
#         if matching_dist == '3d_iou':
#           dets_8corner = [corners(det_tmp) for det_tmp in dets]
#           gts_8corner = [corners(gt_tmp) for gt_tmp in gts]
#           iou_matrix = np.zeros((len(dets_8corner),len(gts_8corner)),dtype=np.float32)
#           for d, det in enumerate(dets_8corner):
#             for g, gt in enumerate(gts_8corner):
#               iou_matrix[d,g] = bbox_iou2d(gt, det, True)
#           distance_matrix = -iou_matrix
#           threshold = -0.1
#         elif matching_dist == '2d_center':
#           distance_matrix = np.zeros((dets.shape[0], gts.shape[0]),dtype=np.float32)
#           for d in range(dets.shape[0]):
#             for g in range(gts.shape[0]):
#               distance_matrix[d][g] = np.sqrt((dets[d][0] - gts[g][0])**2 + (dets[d][1] - gts[g][1])**2)
#           threshold = 2
#         elif matching_dist =='both':
#           dets_8corner = [corners(det_tmp) for det_tmp in dets]
#           gts_8corner = [corners(gt_tmp) for gt_tmp in gts]
#           threshold = 2
#           distance_matrix = np.zeros((dets.shape[0], gts.shape[0]),dtype=np.float32)
#           for d, det in enumerate(dets_8corner):
#             for g, gt in enumerate(gts_8corner):
#               iou_2d = bbox_iou2d(gt, det, True)
#               if iou_2d > 0.1:
#                 distance_matrix[d, g] = np.sqrt((dets[d][0] - gts[g][0])**2 + (dets[d][1] - gts[g][1])**2)
#               else:
#                 distance_matrix[d, g] = threshold + 1.0
#         else:
#           assert(False)
#
#         # GREEDY?
#         # matched_indices = linear_assignment(distance_matrix)
#         matched_indices = greedy_match(distance_matrix, threshold)
#
#         # dets = dets[:, reorder_back]
#         # gts = gts[:, reorder_back]
#         for pair_id in range(matched_indices.shape[0]):
#           if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] < threshold:
#             diff_value = dets[matched_indices[pair_id][0]] - gts[matched_indices[pair_id][1]]
#             diff_value[yaw_pos] = angle_in_range(diff_value[yaw_pos])
#             # x, y, z, yaw, w, l, h
#             diff[tracking_name].append(diff_value)
#
#             gt_track_id = gts_ids[matched_indices[pair_id][1]]
#             if t_idx not in match_diff_t_map[tracking_name]:
#               match_diff_t_map[tracking_name][t_idx] = {gt_track_id: diff_value}
#               match_gt_t_map[tracking_name][t_idx] = {gt_track_id: gts[matched_indices[pair_id][1]][:2]}
#               match_det_t_map[tracking_name][t_idx] = {gt_track_id: dets[matched_indices[pair_id][0]][:2]}
#             else:
#               match_diff_t_map[tracking_name][t_idx][gt_track_id] = diff_value
#               match_gt_t_map[tracking_name][t_idx][gt_track_id] = gts[matched_indices[pair_id][1]][:2]
#               match_det_t_map[tracking_name][t_idx][gt_track_id] = dets[matched_indices[pair_id][0]][:2]
#             # check if we have previous time_step's matching pair for current gt object
#             if t_idx > 0 and t_idx-1 in match_diff_t_map[tracking_name] and gt_track_id in match_diff_t_map[tracking_name][t_idx-1]:
#               det_dist = np.sqrt((dets[matched_indices[pair_id][0]][0] - match_det_t_map[tracking_name][t_idx-1][gt_track_id][0])**2 + (dets[matched_indices[pair_id][0]][1] - match_det_t_map[tracking_name][t_idx-1][gt_track_id][1])**2)
#               gt_dist = np.sqrt((gts[matched_indices[pair_id][1]][0] - match_gt_t_map[tracking_name][t_idx-1][gt_track_id][0])**2 + (gts[matched_indices[pair_id][1]][1] - match_gt_t_map[tracking_name][t_idx-1][gt_track_id][1])**2)
#               diff_dist = det_dist - gt_dist
#               diff_vel_value = diff_value - match_diff_t_map[tracking_name][t_idx-1][gt_track_id]
#               diff_vel_value[yaw_pos] = angle_in_range(diff_vel_value[yaw_pos])
#               diff_vel_value = np.concatenate((diff_vel_value, np.array([diff_dist])))
#
#               diff_vel[tracking_name].append(diff_vel_value)
#
#
#   mean = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
#   std = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
#   var = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
#   mean_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw', w', l', h', v]
#   std_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw', w', l', h', v]
#   var_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw', w', l', h', v]
#   for tracking_name in NUSCENES_TRACKING_NAMES:
#     if len(diff[tracking_name]):
#       diff[tracking_name] = np.stack(diff[tracking_name], axis=0)
#       mean[tracking_name] = np.mean(diff[tracking_name], axis=0)
#       std[tracking_name] = np.std(diff[tracking_name], axis=0)
#       var[tracking_name] = np.var(diff[tracking_name], axis=0)
#     if len(diff_vel[tracking_name]):
#       diff_vel[tracking_name] = np.stack(diff_vel[tracking_name], axis=0)
#       mean_vel[tracking_name] = np.mean(diff_vel[tracking_name], axis=0)
#       std_vel[tracking_name] = np.std(diff_vel[tracking_name], axis=0)
#       var_vel[tracking_name] = np.var(diff_vel[tracking_name], axis=0)
#
#   return mean, std, var, mean_vel, std_vel, var_vel
def read_calib_file(filepath):
  ''' Read in a calibration file and parse into a dictionary.
  Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
  '''
  data = {}
  with open(filepath, 'r') as f:
    for line in f.readlines():
      line = line.rstrip()
      if len(line) == 0: continue
      key, value = line.split(':', 1)
      # The only non-float values in these files are dates, which
      # we don't care about anyway
      try:
        data[key] = np.array([float(x) for x in value.split()])
      except ValueError:
        pass

    # calibs['P2']  = data['P2']
    data['P2'] = np.reshape(data['P2'], [3, 4])
    # Rigid transform from Velodyne coord to reference camera coord
    # self.V2C = data['Tr_velo_to_cam']
    data['Tr_velo_to_cam'] = np.reshape(data['Tr_velo_to_cam'], [3, 4])
    data['Tr_imu_to_velo'] = np.reshape(data['Tr_imu_to_velo'], [3, 4])
    data['R0_rect'] = np.reshape(data['R0_rect'], [3, 3])
    velo_to_cam = np.eye(4, 4)
    imu_to_velo = np.eye(4, 4)
    cam_to_cnt = np.eye(4, 4)
    velo_to_cam[0:3, 0:4] = data['Tr_velo_to_cam']
    imu_to_velo[0:3, 0:4] = data['Tr_imu_to_velo']
    # print(data['R0_rect'])
    cam_to_cnt[0:3, 0:3] = data['R0_rect']
    # velo_to_cam = inverse_rigid_trans(data['Tr_velo_to_cam'])
    # imu_to_velo = inverse_rigid_trans(data['Tr_imu_to_velo'])
    # cnt_to_imu = np.dot(np.dot(np.linalg.inv(imu_to_velo), np.linalg.inv(velo_to_cam)), np.linalg.inv(cam_to_cnt))
    imu_to_cam = np.dot(cam_to_cnt, np.dot(velo_to_cam, imu_to_velo))
    qvelo_to_cam = Quaternion(matrix=velo_to_cam[0:3, 0:3])
    qimu_to_velo = Quaternion(matrix=imu_to_velo[0:3, 0:3])
    qcam_to_cnt = Quaternion(matrix=cam_to_cnt)
    qimu_to_cam = qcam_to_cnt * qvelo_to_cam * qimu_to_velo
  return imu_to_cam, qimu_to_cam  # , qcam_to_imu

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

if __name__ == '__main__':
  tracking_name = 'Pedestrian'
  data_split = 'val'
  d_sha             = 'pointrcnn_' + tracking_name + '_' + data_split
  d_path            = os.path.join("./data//KITTI", d_sha)

  gt_path = './evaluation'
  filename_test_mapping = os.path.join(gt_path, 'evaluate_tracking.seqmap.val')
  # filename_test_mapping = os.path.join(gt_path, 'evaluate_tracking.seqmap.train')
  # data and parameter
  gt_path = os.path.join(gt_path, "label")

  n_frames = []
  sequence_name = []
  with open(filename_test_mapping, "r") as fh:
    for i, l in enumerate(fh):
      fields = l.split(" ")
      sequence_name.append("%04d" % int(fields[0]))
      n_frames.append(int(fields[3]) - int(fields[2]) + 1)

  seq_gt_data = [] # [[] for x in range(len(sequence_name))]
  seq_det_data = []  # [[] for x in range(len(sequence_name))]

  v_state = {class_name: [] for class_name in KITTI_CLASS_NAMES}
  a_state = {class_name: [] for class_name in KITTI_CLASS_NAMES}

  diff = {tracking_name: [] for tracking_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  diff_vel = {tracking_name: [] for tracking_name in KITTI_CLASS_NAMES} # [x', y', z', yaw']
  for seq, s_name in enumerate(sequence_name):
    gt_filename = os.path.join(gt_path, "%s.txt" % s_name)
    gt_f = open(gt_filename, "r")

    # gt_data         = [[] for x in range(n_frames[seq])] # current set has only 1059 entries, sufficient length is checked anyway

    d_filename = os.path.join(d_path, "%s.txt" % s_name)
    d_f = open(d_filename, "r")
    d_data         = [[] for x in range(n_frames[seq])] # current set has only 1059 entries, sufficient length is checked anyway


    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    # det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    oxts_file = os.path.join('./data/KITTI/resources/training/oxts', s_name + '.txt')
    calib_file = os.path.join('./data/KITTI/resources/training/calib', s_name + '.txt')
    seq_oxts = load_oxts_packets_and_poses(oxts_files=[oxts_file])
    imu_to_cam, qimu_to_cam = read_calib_file(calib_file)
    cam_to_imu = np.linalg.inv(imu_to_cam)
    qcam_to_imu = qimu_to_cam.inverse

    gt_data = {class_name: [] for class_name in KITTI_CLASS_NAMES}

    # seq_gts = np.loadtxt(gt_filename, delimiter=' ')  # load detections, N x 15
    for line in gt_f:
      # KITTI tracking benchmark data format:
      line = line.strip()
      fields = line.split(" ")
      frame = int(float(fields[0]))  # frame
      track_id = int(float(fields[1]))  # id
      obj_type = fields[2]#.lower()  # object type [car, pedestrian, cyclist, ...]
      truncation = int(float(fields[3]))  # truncation [-1,0,1,2]
      occlusion = int(float(fields[4]))  # occlusion  [-1,0,1,2]
      h = float(fields[10])  # height [m]
      w = float(fields[11])  # width  [m]
      l = float(fields[12])  # length [m]
      pos_x = float(fields[13])  # X [m]
      pos_y = float(fields[14])  # Y [m]
      pos_z = float(fields[15])  # Z [m]
      yaw = float(fields[16])  # yaw angle [rad]
      # do not consider objects marked as invalid
      if obj_type not in KITTI_CLASS_NAMES:
        continue
      if track_id is -1 and obj_type != "dontcare":
        continue

      loc_pose = np.array([[pos_x, pos_y, pos_z]]) # dets[:, 3:6].copy()
      cam_to_world = np.dot(seq_oxts[frame].T_w_imu, cam_to_imu)  # 4x4
      qimu_to_world = Quaternion(matrix=seq_oxts[frame].T_w_imu[0:3, 0:3])
      qworld_to_imu = qimu_to_world.inverse

      qlyaw = Quaternion(axis=[0, 1, 0], radians=yaw)
      qgyaw = qimu_to_world * qcam_to_imu * qlyaw
      rpy = quaternion_to_euler(qgyaw)
      glo_pose = np.dot(cam_to_world[0:3, 0:3], loc_pose.T) + cam_to_world[0:3, 3].reshape(-1, 1)
      gpos = np.transpose(glo_pose[:, 0])
      gyaw = rpy[2]
      gt_data[obj_type].append([int(track_id), int(frame), gpos[0], gpos[1], gpos[2], gyaw, w, l, h])

      # print(yaw, gyaw)
    gt_f.close()
    seq_gt_data.append(gt_data)

    det_data = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    for class_name in KITTI_CLASS_NAMES:
      d_sha = 'pointrcnn_' + class_name + '_' + data_split
      d_path = os.path.join("./data/KITTI", d_sha)
      d_filename = os.path.join(d_path, "%s.txt" % s_name)
      seq_dets = np.loadtxt(d_filename, delimiter=',')  # load detections, N x 15
      # print(seq_dets)
      for det in seq_dets:
        loc_pose = np.array([[det[10], det[11], det[12]]]) # dets[:, 3:6].copy()
        cam_to_world = np.dot(seq_oxts[int(det[0])].T_w_imu, cam_to_imu)  # 4x4
        qimu_to_world = Quaternion(matrix=seq_oxts[int(det[0])].T_w_imu[0:3, 0:3])
        qworld_to_imu = qimu_to_world.inverse
        qlyaw = Quaternion(axis=[0, 1, 0], radians=det[13])
        qgyaw = qimu_to_world * qcam_to_imu * qlyaw
        rpy = quaternion_to_euler(qgyaw)
        glo_pose = np.dot(cam_to_world[0:3, 0:3], loc_pose.T) + cam_to_world[0:3, 3].reshape(-1, 1)
        gpos = np.transpose(glo_pose[:, 0])
        gyaw = rpy[2]
        det_data[class_name].append([-1, int(det[0]), gpos[0], gpos[1], gpos[2], gyaw, det[8], det[9], det[7]])

        # print(det[13], gyaw)
    seq_det_data.append(det_data)

  # Seq



  # Load All Dataset from Detection and GroundTruth
  # # Process Noise
  # for seq, seq_gts in enumerate(seq_gt_data):
    for class_name in KITTI_CLASS_NAMES:
      sgts = sorted(gt_data[class_name])
      for idx, gt in enumerate(sgts):
        id = gt[0]
        frame = gt[1]
        next_frame = frame + 1
        if idx + 1 < len(sgts):
          if next_frame == sgts[idx + 1][1] and id == sgts[idx + 1][0]:
            t0_state = np.array(sgts[idx][2:9])
            t1_state = np.array(sgts[idx+1][2:9])
            diff1 = t1_state - t0_state
            dist1 = np.sqrt((diff1[0])**2 + (diff1[1])**2)
            v_state[class_name].append(np.concatenate((diff1, np.array([dist1]))))
            next_next_frame = next_frame + 1
            if idx + 2 < len(sgts):
              if next_next_frame == sgts[idx + 2][1] and id == sgts[idx + 2][0]:
                t2_state = np.array(sgts[idx+2][2:9])
                diff2 = t2_state - t1_state
                dist2 = np.sqrt((diff2[0])**2 + (diff2[1])**2)
                diff_diff = diff2 - diff1
                diff_dist = dist2 - dist1
                a_state[class_name].append(np.concatenate((diff_diff, np.array([diff_dist]))))

    # Measurement Noise
    # id, frame, x, y, z, yaw, w, l, h
    match_diff_t_map = {tracking_name: {} for tracking_name in KITTI_CLASS_NAMES}
    match_gt_t_map = {tracking_name: {} for tracking_name in KITTI_CLASS_NAMES}
    match_det_t_map = {tracking_name: {} for tracking_name in KITTI_CLASS_NAMES}
    for class_name in KITTI_CLASS_NAMES:
      if len(det_data[class_name]) == 0 or len(gt_data[class_name]) == 0:
        continue
      np_det = np.stack(det_data[class_name],  axis=0)
      np_gt = np.stack(gt_data[class_name],  axis=0)
      for frame in range(n_frames[seq]):
        dets = np_det[np_det[:, 1] == frame, :]
        gts = np_gt[np_gt[:, 1] == frame, :]
        distance_matrix = np.zeros((dets.shape[0], gts.shape[0]), dtype=np.float32)
        for d in range(dets.shape[0]):
          for g in range(gts.shape[0]):
            distance_matrix[d, g] = np.sqrt((dets[d, 2] - gts[g, 2])**2 + (dets[d, 3] - gts[g, 3])**2)
        if class_name == "Car":
          threshold = 1.0
        elif class_name == "Cyclist":
          threshold = 0.5
        elif class_name == "Pedestrian":
          threshold = 0.5
        yaw_pos = 3
        matched_indices = greedy_match(distance_matrix, threshold)

        for pair_id in range(matched_indices.shape[0]):
          if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] < threshold:
            diff_value = dets[matched_indices[pair_id][0], 2:] - gts[matched_indices[pair_id][1], 2:]
            diff_value[yaw_pos] = angle_in_range(diff_value[yaw_pos])
            # x, y, z, yaw, w, l, h
            diff[class_name].append(diff_value)

            gt_track_id = gts[matched_indices[pair_id][1], 0]
            if frame not in match_diff_t_map[class_name]:
              match_diff_t_map[class_name][frame] = {gt_track_id: diff_value}
              match_gt_t_map[class_name][frame] = {gt_track_id: gts[matched_indices[pair_id][1], 2:]}
              match_det_t_map[class_name][frame] = {gt_track_id: dets[matched_indices[pair_id][0], 2:]}
            else:
              match_diff_t_map[class_name][frame][gt_track_id] = diff_value
              match_gt_t_map[class_name][frame][gt_track_id] = gts[matched_indices[pair_id][1], 2:]
              match_det_t_map[class_name][frame][gt_track_id] = dets[matched_indices[pair_id][0], 2:]

            # check if we have previous time_step's matching pair for current gt object
            if frame > 0 and frame-1 in match_diff_t_map[class_name] and gt_track_id in match_diff_t_map[class_name][frame-1]:
              det_dist = np.sqrt((dets[matched_indices[pair_id][0], 2] - match_det_t_map[class_name][frame-1][gt_track_id][0])**2 + (dets[matched_indices[pair_id][0], 3] - match_det_t_map[class_name][frame-1][gt_track_id][1])**2)
              gt_dist = np.sqrt((gts[matched_indices[pair_id][1], 2] - match_gt_t_map[class_name][frame-1][gt_track_id][0])**2 + (gts[matched_indices[pair_id][1], 3] - match_gt_t_map[class_name][frame-1][gt_track_id][1])**2)
              diff_dist = det_dist - gt_dist
              diff_vel_value = diff_value - match_diff_t_map[class_name][frame-1][gt_track_id]
              diff_vel_value[yaw_pos] = angle_in_range(diff_vel_value[yaw_pos])
              diff_vel_value = np.concatenate((diff_vel_value, np.array([diff_dist])))

              diff_vel[class_name].append(diff_vel_value)


  R_mean = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  R_std = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  R_var = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  R_mean_vel = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x', y', z', yaw', w', l', h', v]
  R_std_vel = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x', y', z', yaw', w', l', h', v]
  R_var_vel = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x', y', z', yaw', w', l', h', v]
  for class_name in KITTI_CLASS_NAMES:
    if len(diff[class_name]):
      diff[class_name] = np.stack(diff[class_name], axis=0)
      R_mean[class_name] = np.mean(diff[class_name], axis=0)
      R_std[class_name] = np.std(diff[class_name], axis=0)
      R_var[class_name] = np.var(diff[class_name], axis=0)
    if len(diff_vel[class_name]):
      diff_vel[class_name] = np.stack(diff_vel[class_name], axis=0)
      R_mean_vel[class_name] = np.mean(diff_vel[class_name], axis=0)
      R_std_vel[class_name] = np.std(diff_vel[class_name], axis=0)
      R_var_vel[class_name] = np.var(diff_vel[class_name], axis=0)



  Q_mean = {class_name: [] for class_name in KITTI_CLASS_NAMES}
  Q_std = {class_name: [] for class_name in KITTI_CLASS_NAMES}
  Q_var = {class_name: [] for class_name in KITTI_CLASS_NAMES}
  for class_name in KITTI_CLASS_NAMES:
    if len(v_state[class_name]):
      v_state[class_name] = np.stack(v_state[class_name], axis=0)
      v_mean = np.mean(v_state[class_name], axis=0)
      v_std = np.std(v_state[class_name], axis=0)
      v_var = np.var(v_state[class_name], axis=0)
    if len(a_state[class_name]):
      a_state[class_name] = np.stack(a_state[class_name], axis=0)
      a_mean = np.mean(a_state[class_name], axis=0)
      a_std = np.std(a_state[class_name], axis=0)
      a_var = np.var(a_state[class_name], axis=0)

    Q_mean[class_name] = np.hstack((v_mean, a_mean))
    Q_std[class_name] = np.hstack((v_std, a_std))
    Q_var[class_name] = np.hstack((v_var, a_var))

    m_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    m_no_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    a_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    for class_name in KITTI_CLASS_NAMES:
      if not len(Q_var[class_name]):
        continue
      m_head_Q[class_name] = [Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11],
                                 Q_var[class_name][15], Q_var[class_name][11]]
      m_no_head_Q[class_name] = [Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11],
                                    Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11]]
      a_Q[class_name] = [Q_var[class_name][11], Q_var[class_name][10], 0.0, 0.0, 0.0, Q_var[class_name][10]]

    m_R = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    m_head_P = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    m_no_head_P = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    a_P = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    a_R = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    for class_name in KITTI_CLASS_NAMES:
      if not len(R_var[class_name]):
        continue
      m_R[class_name] = [R_var[class_name][0], R_var[class_name][1], R_var[class_name][3]]
      m_head_P[class_name] = [R_var[class_name][0], R_var[class_name][1], R_var[class_name][3],
                                 R_var_vel[class_name][7], R_var_vel[class_name][3]]
      m_no_head_P[class_name] = [R_var[class_name][0], R_var[class_name][1], R_var[class_name][3],
                                    R_var_vel[class_name][0], R_var_vel[class_name][1], R_var_vel[class_name][3]]
      a_P[class_name] = [R_var[class_name][3], R_var[class_name][2], R_var[class_name][4], R_var[class_name][5], R_var[class_name][6],
                            R_var_vel[class_name][2]]
      a_R[class_name] = [R_var[class_name][3], R_var[class_name][2], R_var[class_name][4], R_var[class_name][5], R_var[class_name][6]]


    # m_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    # m_no_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    # a_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    # for class_name in KITTI_CLASS_NAMES:
    #   if not len(Q_var[class_name]):
    #     continue
    #   m_head_Q[class_name] = [Q_var[class_name][0], Q_var[class_name][1], Q_var[class_name][3],
    #                              Q_var[class_name][15], Q_var[class_name][11]]
    #   m_no_head_Q[class_name] = [Q_var[class_name][0], Q_var[class_name][1], Q_var[class_name][3],
    #                                 Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11]]
    #   a_Q[class_name] = [Q_var[class_name][3], Q_var[class_name][2], 0.0, 0.0, 0.0, Q_var[class_name][10]]
    #
    # m_R = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    # m_head_P = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    # m_no_head_P = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    # a_P = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    # a_R = {class_name: [] for class_name in KITTI_CLASS_NAMES}  # [x, y, z, yaw, w, l, h]
    # for class_name in KITTI_CLASS_NAMES:
    #   if not len(R_var[class_name]):
    #     continue
    #   m_R[class_name] = [R_var[class_name][0], R_var[class_name][1], R_var[class_name][3]]
    #   m_head_P[class_name] = [R_var[class_name][0], R_var[class_name][1], R_var[class_name][3],
    #                              R_var_vel[class_name][7], R_var_vel[class_name][3]]
    #   m_no_head_P[class_name] = [R_var[class_name][0], R_var[class_name][1], R_var[class_name][3],
    #                                 R_var_vel[class_name][0], R_var_vel[class_name][1], R_var_vel[class_name][3]]
    #   a_P[class_name] = [R_var[class_name][3], R_var[class_name][2], R_var[class_name][4], R_var[class_name][5], R_var[class_name][6],
    #                         R_var_vel[class_name][2]]
    #   a_R[class_name] = [R_var[class_name][3], R_var[class_name][2], R_var[class_name][4], R_var[class_name][5], R_var[class_name][6]]

  print('m_head_P = ', m_head_P)
  print('m_head_Q = ', m_head_Q)
  print('m_no_head_P = ', m_no_head_P)
  print('m_no_head_Q = ', m_no_head_Q)
  print('m_R = ', m_R)
  print('a_P = ', a_P)
  print('a_Q = ', a_Q)
  print('a_R = ', a_R)