import os
import sys

import numpy as np
# from utils import iou3d, convert_3dbox_to_8corner
from sklearn.utils.linear_assignment_ import linear_assignment

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.loaders import create_tracks
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from utils import angle_in_range, corners, bbox_iou2d, bbox_iou3d, bbox_adjacency_2d

import argparse

NUSCENES_TRACKING_NAMES = [
  'bicycle',
  'bus',
  'car',
  'motorcycle',
  'pedestrian',
  'trailer',
  'truck'
]

def rotation_to_positive_z_angle(rotation):
  q = Quaternion(rotation)
  angle = q.angle if q.axis[2] > 0 else -q.angle
  return angle

def get_mean(tracks):
  '''
  Input:
    tracks: {scene_token:  {t: [TrackingBox]}}
  '''
  print('len(tracks.keys()): ', len(tracks.keys()))
  yaw_pos = 3
  # x, y, yaw, z, w, l, h, x', y', yaw', z', w', l', h'
  gt_trajectory_map = {tracking_name: {scene_token: {} for scene_token in tracks.keys()} for tracking_name in NUSCENES_TRACKING_NAMES}

  # store every detection data to compute mean and variance
  gt_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
  # no_head_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # = np.array([0, 0, 0, 0])  # x, y, x', y'
  # head_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # = np.array([0, 0, 0, 0, 0])  # x, y', yaw, v, yaw'
  # appearance_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # = np.array([0, 0, 0, 0, 0])  # , z, w, l, h, z'

  for scene_token in tracks.keys():
    # print('scene_token: ', scene_token)
    # print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
    for t_idx in range(len(tracks[scene_token].keys())):
      #print('t_idx: ', t_idx)
      t = sorted(tracks[scene_token].keys())[t_idx]
      # print(tracks[scene_token][t])
      for box_id, box in enumerate(tracks[scene_token][t]): # range(len(tracks[scene_token][t])):
        #print('box_id: ', box_id)
        # box = tracks[scene_token][t][box_id]
        #print('box: ', box)
       
        if box.tracking_name not in NUSCENES_TRACKING_NAMES:
          continue
        # x, y, z, yaw, x', y', z', yaw', v, x", y", z", yaw", a
        yaw = quaternion_yaw(Quaternion(box.rotation))
        box_data = np.array([box.translation[0], box.translation[1], box.translation[2], yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        box_data[yaw_pos] = angle_in_range(box_data[yaw_pos])
        if box.tracking_id not in gt_trajectory_map[box.tracking_name][scene_token]:
          gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id] = {t_idx: box_data}
        else: 
          gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data

        # if we can find the same object in the previous frame, get the velocity
        if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx-1 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
          prev_box_data = gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx-1]
          dist = np.sqrt((box_data[0] - prev_box_data[0])**2 + (box_data[1] - prev_box_data[1])**2)
          residual_vel = box_data[:4] - prev_box_data[:4] # x, y, yaw
          # angle in range
          residual_vel[yaw_pos] = angle_in_range(residual_vel[yaw_pos])
          box_data[4:8] = residual_vel
          box_data[8] = dist
          gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
          # if we can find the same object in the previous two frames, get the acceleration
          if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx-2 in gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
            pprev_box_data = gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 2]
            pdist = np.sqrt((prev_box_data[0] - pprev_box_data[0])**2 + (prev_box_data[1] - pprev_box_data[1])**2)
            presidual_vel = prev_box_data[:4] - pprev_box_data[:4] # x, y, yaw
            presidual_vel[yaw_pos] = angle_in_range(presidual_vel[yaw_pos])
            residual_a = residual_vel - presidual_vel
            residual_a[yaw_pos] = angle_in_range(residual_a[yaw_pos])
            dist_a = dist - pdist
            box_data[9:13] = residual_a
            box_data[13] = dist_a
            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
            gt_box_data[box.tracking_name].append(box_data)
            # # x, y, z, yaw, x', y', z', yaw', v, x", y", z", yaw", a
            # no_head_box_data[box.tracking_name].append(np.array([box_data[0], box_data[1], box_data[3], box_data[4], box_data[5], box_data[7]]))
            # head_box_data[box.tracking_name].append(np.array([box_data[0], box_data[1], box_data[3], box_data[8], box_data[7]]))
            # appearance_box_data[box.tracking_name].append(np.array([box_data[2], 0.0, 0.0, 0.0, box_data[6]]))

  mean = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  std = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  var = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  for tracking_name in NUSCENES_TRACKING_NAMES:
    if len(gt_box_data[tracking_name]):
      gt_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)
      # no_head_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)
      # head_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)
      # appearance_box_data[tracking_name] = np.stack(gt_box_data[tracking_name], axis=0)

      mean[tracking_name] = np.mean(gt_box_data[tracking_name], axis=0)
      std[tracking_name] = np.std(gt_box_data[tracking_name], axis=0)
      var[tracking_name] = np.var(gt_box_data[tracking_name], axis=0)
  #
  # # x, y, z, yaw, x', y', z', yaw', v, x", y", z", yaw", a
  # mean = {tracking_name: np.mean(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  # std = {tracking_name: np.std(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
  # var = {tracking_name: np.var(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}


  # no_head_mean = {tracking_name: np.mean(no_head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  # no_head_std = {tracking_name: np.std(no_head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  # no_head_var = {tracking_name: np.var(no_head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  #
  #
  # head_mean = {tracking_name: np.mean(head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  # head_std = {tracking_name: np.std(head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  # head_var = {tracking_name: np.var(head_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  #
  #
  # appearance_mean = {tracking_name: np.mean(appearance_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  # appearance_std = {tracking_name: np.std(appearance_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  # appearance_var = {tracking_name: np.var(appearance_box_data[tracking_name], axis=0).tolist() for tracking_name in NUSCENES_TRACKING_NAMES}
  return mean, std, var #, head_std, head_var, appearance_mean, appearance_std, appearance_var


def matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks_gt, matching_dist):
  '''
  For each sample token, find matches of pred_boxes and gt_boxes, then get stats.
  tracks_gt has the temporal order info for each sample_token
  '''
  yaw_pos = 3
  diff = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  diff_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw']

  # # similar to main.py class AB3DMOT update()
  # reorder = [3, 4, 5, 6, 2, 1, 0]
  # reorder_back = [6, 5, 4, 0, 1, 2, 3]

  for scene_token in tracks_gt.keys():
    #print('scene_token: ', scene_token)
    #print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
    # {tracking_name: t_idx: tracking_id: det(7) }
    match_diff_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
    match_gt_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
    match_det_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
    for t_idx in range(len(tracks_gt[scene_token].keys())):
      #print('t_idx: ', t_idx)
      t = sorted(tracks_gt[scene_token].keys())[t_idx]
      #print(len(tracks_gt[scene_token][t]))
      if len(tracks_gt[scene_token][t]) == 0:
        continue
      ref_box = tracks_gt[scene_token][t][0]
      sample_token = ref_box.sample_token

      for tracking_name in NUSCENES_TRACKING_NAMES:
        #print('t: ', t)
        gt_all = [box for box in gt_boxes.boxes[sample_token] if box.tracking_name == tracking_name]
        if len(gt_all) == 0:
          continue

        gts = np.stack([np.array([box.translation[0], box.translation[1], box.translation[2], angle_in_range(quaternion_yaw(Quaternion(box.rotation))), box.size[0], box.size[1], box.size[2]]) for box in gt_all], axis=0)


        gts_ids = [box.tracking_id for box in gt_all]
        # gts = np.stack([np.array([
        #   box.size[2], box.size[0], box.size[1],
        #   box.translation[0], box.translation[1], box.translation[2],
        #   rotation_to_positive_z_angle(box.rotation)
        #   ]) for box in gt_all], axis=0)
        # gts_ids = [box.tracking_id for box in gt_all]

        det_all = [box for box in pred_boxes.boxes[sample_token] if box.detection_name == tracking_name]
        if len(det_all) == 0:
          continue

        dets = np.stack([np.array([box.translation[0], box.translation[1], box.translation[2], angle_in_range(quaternion_yaw(Quaternion(box.rotation))), box.size[0], box.size[1], box.size[2]]) for box in det_all], axis=0)

        #
        # dets = dets[:, reorder]
        # gts = gts[:, reorder]

        if matching_dist == '3d_iou':
          dets_8corner = [corners(det_tmp) for det_tmp in dets]
          gts_8corner = [corners(gt_tmp) for gt_tmp in gts]
          iou_matrix = np.zeros((len(dets_8corner),len(gts_8corner)),dtype=np.float32)
          for d, det in enumerate(dets_8corner):
            for g, gt in enumerate(gts_8corner):
              iou_matrix[d,g] = bbox3d_iou3d(det, gt)[0]
          distance_matrix = -iou_matrix
          threshold = -0.5
        elif matching_dist == '2d_center':
          distance_matrix = np.zeros((dets.shape[0], gts.shape[0]),dtype=np.float32)
          for d in range(dets.shape[0]):
            for g in range(gts.shape[0]):
              distance_matrix[d][g] = np.sqrt((dets[d][0] - gts[g][0])**2 + (dets[d][1] - gts[g][1])**2) 
          threshold = 2
        else:
          assert(False) 

        # GREEDY?
        matched_indices = linear_assignment(distance_matrix)

        # dets = dets[:, reorder_back]
        # gts = gts[:, reorder_back]
        for pair_id in range(matched_indices.shape[0]):
          if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] < threshold:
            diff_value = dets[matched_indices[pair_id][0]] - gts[matched_indices[pair_id][1]]
            diff_value[yaw_pos] = angle_in_range(diff_value[yaw_pos])
            # x, y, z, yaw, w, l, h
            diff[tracking_name].append(diff_value)

            gt_track_id = gts_ids[matched_indices[pair_id][1]]
            if t_idx not in match_diff_t_map[tracking_name]:
              match_diff_t_map[tracking_name][t_idx] = {gt_track_id: diff_value}
              match_gt_t_map[tracking_name][t_idx] = {gt_track_id: gts[matched_indices[pair_id][1]][:2]}
              match_det_t_map[tracking_name][t_idx] = {gt_track_id: dets[matched_indices[pair_id][0]][:2]}
            else:
              match_diff_t_map[tracking_name][t_idx][gt_track_id] = diff_value
              match_gt_t_map[tracking_name][t_idx][gt_track_id] = gts[matched_indices[pair_id][1]][:2]
              match_det_t_map[tracking_name][t_idx][gt_track_id] = dets[matched_indices[pair_id][0]][:2]
            # check if we have previous time_step's matching pair for current gt object
            if t_idx > 0 and t_idx-1 in match_diff_t_map[tracking_name] and gt_track_id in match_diff_t_map[tracking_name][t_idx-1]:
              det_dist = np.sqrt((dets[matched_indices[pair_id][0]][0] - match_det_t_map[tracking_name][t_idx-1][gt_track_id][0])**2 + (dets[matched_indices[pair_id][0]][1] - match_det_t_map[tracking_name][t_idx-1][gt_track_id][1])**2)
              gt_dist = np.sqrt((gts[matched_indices[pair_id][1]][0] - match_gt_t_map[tracking_name][t_idx-1][gt_track_id][0])**2 + (gts[matched_indices[pair_id][1]][1] - match_gt_t_map[tracking_name][t_idx-1][gt_track_id][1])**2)
              diff_dist = det_dist - gt_dist
              diff_vel_value = diff_value - match_diff_t_map[tracking_name][t_idx-1][gt_track_id]
              diff_vel_value[yaw_pos] = angle_in_range(diff_vel_value[yaw_pos])
              diff_vel_value = np.concatenate((diff_vel_value, np.array([diff_dist])))

              diff_vel[tracking_name].append(diff_vel_value)


  mean = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  std = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  var = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  mean_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw', w', l', h', v]
  std_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw', w', l', h', v]
  var_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x', y', z', yaw', w', l', h', v]
  for tracking_name in NUSCENES_TRACKING_NAMES:
    if len(diff[tracking_name]):
      diff[tracking_name] = np.stack(diff[tracking_name], axis=0)
      mean[tracking_name] = np.mean(diff[tracking_name], axis=0)
      std[tracking_name] = np.std(diff[tracking_name], axis=0)
      var[tracking_name] = np.var(diff[tracking_name], axis=0)
    if len(diff_vel[tracking_name]):
      diff_vel[tracking_name] = np.stack(diff_vel[tracking_name], axis=0)
      mean_vel[tracking_name] = np.mean(diff_vel[tracking_name], axis=0)
      std_vel[tracking_name] = np.std(diff_vel[tracking_name], axis=0)
      var_vel[tracking_name] = np.var(diff_vel[tracking_name], axis=0)

  return mean, std, var, mean_vel, std_vel, var_vel

if __name__ == '__main__':
  # Settings.
  parser = argparse.ArgumentParser(description='Get nuScenes stats.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--eval_set', type=str, default='train',
                      help='Which dataset split to evaluate on, train, val or test.')
  parser.add_argument('--config_path', type=str, default='',
                      help='Path to the configuration file.'
                           'If no path given, the NIPS 2019 configuration will be used.')
  parser.add_argument('--verbose', type=int, default=1,
                      help='Whether to print to stdout.')
  parser.add_argument('--matching_dist', type=str, default='2d_center',
                      help='Which distance function for matching, 3d_iou or 2d_center.')
  args = parser.parse_args()

  eval_set_ = args.eval_set
  config_path = args.config_path
  verbose_ = bool(args.verbose)
  matching_dist = args.matching_dist

  if config_path == '':
    cfg_ = config_factory('tracking_nips_2019')
  else:
    with open(config_path, 'r') as _f:
      cfg_ = DetectionConfig.deserialize(json.load(_f))

  if 'train' == eval_set_:
    detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_train.json'
    data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-trainval'
    version = 'v1.0-trainval'
  elif 'val' == eval_set_:
    detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_val.json'
    data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-trainval'
    version = 'v1.0-trainval'
  elif 'mini_val' == eval_set_:
    detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_mini_val.json'
    data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-mini'
    version = 'v1.0-mini'
  elif 'mini_train' == eval_set_:
    detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/detection/megvii_mini_train.json'
    data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/nuscenes/v1.0-mini'
    version = 'v1.0-mini'

  nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

  pred_boxes, _ = load_prediction(detection_file, 10000, DetectionBox)
  gt_boxes = load_gt(nusc, eval_set_, TrackingBox)

  assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

  # Add center distances.
  pred_boxes = add_center_dist(nusc, pred_boxes)
  gt_boxes = add_center_dist(nusc, gt_boxes)

  
  print('len(pred_boxes.sample_tokens): ', len(pred_boxes.sample_tokens))
  print('len(gt_boxes.sample_tokens): ', len(gt_boxes.sample_tokens))

  tracks_gt = create_tracks(gt_boxes, nusc, eval_set_, gt=True)

  gt_mean, gt_std, gt_var = get_mean(tracks_gt)
  print('Q: from relationship between GT')
  print('x, y, z, yaw, dx, dy, dz, dyaw, v, ddx, ddy, ddz, ddyaw, a')
  print('gt_mean = ', gt_mean)
  print('gt_std = ', gt_std)
  print('gt_var = ', gt_var)

  m_head_Q = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
  m_no_head_Q = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
  a_Q = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
  for tracking_name in NUSCENES_TRACKING_NAMES:
    if not len(gt_var[tracking_name]):
      continue
    m_head_Q[tracking_name] = [gt_var[tracking_name][9], gt_var[tracking_name][10], gt_var[tracking_name][12], gt_var[tracking_name][13], gt_var[tracking_name][12]]
    m_no_head_Q[tracking_name] = [gt_var[tracking_name][9], gt_var[tracking_name][10], gt_var[tracking_name][12], gt_var[tracking_name][9], gt_var[tracking_name][10], gt_var[tracking_name][12]]
    a_Q[tracking_name] = [gt_var[tracking_name][11], 0.0, 0.0, 0.0, gt_var[tracking_name][11]]

  # for observation noise covariance
  mean, std, var, mean_vel, std_vel, var_vel = matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks_gt, matching_dist)

  print('R & P: from relationship between Det and GT')
  print('x, y, z, yaw, w, l, h')
  print('mean = ', mean)
  print('std = ', std)
  print('var = ', var)
  print('dx, dy, dz, dyaw, dw, dl, dh, v')
  print('mean_vel = ', mean_vel)
  print('std_vel = ', std_vel)
  print('var_vel = ', var_vel)

  m_R = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  m_head_P = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  m_no_head_P = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  a_P = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  a_R = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES} # [x, y, z, yaw, w, l, h]
  for tracking_name in NUSCENES_TRACKING_NAMES:
    if not len(var[tracking_name]):
      continue
    m_R[tracking_name] = [var[tracking_name][0], var[tracking_name][1], var[tracking_name][3]]
    m_head_P[tracking_name] = [var[tracking_name][0], var[tracking_name][1], var[tracking_name][3], var_vel[tracking_name][7], var_vel[tracking_name][3]]
    m_no_head_P[tracking_name] = [var[tracking_name][0], var[tracking_name][1], var[tracking_name][3], var_vel[tracking_name][1], var_vel[tracking_name][2], var_vel[tracking_name][3]]
    a_P[tracking_name] = [var[tracking_name][2], var[tracking_name][4], var[tracking_name][5], var[tracking_name][6], var_vel[tracking_name][2]]
    a_R[tracking_name] = [var[tracking_name][2], var[tracking_name][4], var[tracking_name][5], var[tracking_name][6]]

  print('m_head_P = ', m_head_P)
  print('m_head_Q = ', m_head_Q)
  print('m_no_head_P = ', m_no_head_P)
  print('m_no_head_Q = ', m_no_head_Q)
  print('m_R = ', m_R)
  print('a_P = ', a_P)
  print('a_Q = ', a_Q)
  print('a_R = ', a_R)