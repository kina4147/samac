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

from matplotlib import pyplot as plt

import os.path, copy, numpy as np, time, sys
from pykitti.utils import load_oxts_packets_and_poses

from SAMAC3DMOT.utils import angle_in_range, quaternion_to_euler
import argparse

KITTI_CLASS_NAMES = [
  'Car',
  'Pedestrian',
  'Cyclist'
]

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
  # filename_test_mapping = os.path.join(gt_path, 'evaluate_tracking.seqmap.val')
  filename_test_mapping = os.path.join(gt_path, 'evaluate_tracking.seqmap.train')
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

  diff = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  ego_dist = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  ego_angle = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  gt_crowd = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  det_angle = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x, y, z, yaw, w, l, h]
  diff_vel = {class_name: [] for class_name in KITTI_CLASS_NAMES} # [x', y', z', yaw']
  for seq, s_name in enumerate(sequence_name):
    gt_filename = os.path.join(gt_path, "%s.txt" % s_name)
    gt_f = open(gt_filename, "r")
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
      obs_angle = float(fields[5])          # observation angle [rad]
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
      tyaw = angle_in_range(yaw + np.arctan2(-pos_x, pos_z))
      gt_data[obj_type].append([int(track_id), int(frame), gpos[0], gpos[1], gpos[2], gyaw, w, l, h, np.sqrt(pos_x * pos_x + pos_z + pos_z), tyaw])
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
        tyaw = angle_in_range(det[13] + np.arctan2(-det[10], det[12]))
        det_data[class_name].append([-1, int(det[0]), gpos[0], gpos[1], gpos[2], gyaw, det[8], det[9], det[7], np.sqrt(det[10] * det[10] + det[12] + det[12]), tyaw])
    seq_det_data.append(det_data)


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
            diff1[3] = angle_in_range(diff1[3])
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
        if len(dets) == 0 or len(gts) == 0:
          continue
        distance_matrix = np.zeros((dets.shape[0], gts.shape[0]), dtype=np.float32)
        for d in range(dets.shape[0]):
          for g in range(gts.shape[0]):
            distance_matrix[d, g] = np.sqrt((dets[d, 2] - gts[g, 2])**2 + (dets[d, 3] - gts[g, 3])**2)
        if class_name == "Car":
          threshold = 2.0
        elif class_name == "Cyclist":
          threshold = 0.5
        elif class_name == "Pedestrian":
          threshold = 0.5
        yaw_pos = 3
        distance_matrix[distance_matrix > threshold] = 0.0
        crowd = np.count_nonzero(distance_matrix, axis=0)

        matched_indices = greedy_match(distance_matrix, threshold)

        for pair_id in range(matched_indices.shape[0]):
          if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] < threshold:
            dist = gts[matched_indices[pair_id][1], 9]
            angle = gts[matched_indices[pair_id][1], 10]
            dangle = dets[matched_indices[pair_id][0], 10]
            gcrowd = crowd[matched_indices[pair_id][1]]
            diff_value = dets[matched_indices[pair_id][0], 2:9] - gts[matched_indices[pair_id][1], 2:9]
            diff_value[yaw_pos] = angle_in_range(diff_value[yaw_pos])
            if abs(diff_value[yaw_pos]) > np.pi - 0.34: # reverse
              if diff_value[yaw_pos] > 0:
                diff_value[yaw_pos] = diff_value[yaw_pos] - np.pi
              elif diff_value[yaw_pos] < 0:
                diff_value[yaw_pos] = diff_value[yaw_pos] + np.pi
            # x, y, z, yaw, w, l, h
            diff[class_name].append(diff_value)
            ego_dist[class_name].append(dist)
            ego_angle[class_name].append(angle)
            det_angle[class_name].append(dangle)
            gt_crowd[class_name].append(gcrowd)
            gt_track_id = gts[matched_indices[pair_id][1], 0]
            if frame not in match_diff_t_map[class_name]:
              match_diff_t_map[class_name][frame] = {gt_track_id: diff_value}
              match_gt_t_map[class_name][frame] = {gt_track_id: gts[matched_indices[pair_id][1], 2:9]}
              match_det_t_map[class_name][frame] = {gt_track_id: dets[matched_indices[pair_id][0], 2:9]}
            else:
              match_diff_t_map[class_name][frame][gt_track_id] = diff_value
              match_gt_t_map[class_name][frame][gt_track_id] = gts[matched_indices[pair_id][1], 2:9]
              match_det_t_map[class_name][frame][gt_track_id] = dets[matched_indices[pair_id][0], 2:9]

            # check if we have previous time_step's matching pair for current gt object
            if frame > 0 and frame-1 in match_diff_t_map[class_name] and gt_track_id in match_diff_t_map[class_name][frame-1]:
              det_dist = np.sqrt((dets[matched_indices[pair_id][0], 2] - match_det_t_map[class_name][frame-1][gt_track_id][0])**2 + (dets[matched_indices[pair_id][0], 3] - match_det_t_map[class_name][frame-1][gt_track_id][1])**2)
              gt_dist = np.sqrt((gts[matched_indices[pair_id][1], 2] - match_gt_t_map[class_name][frame-1][gt_track_id][0])**2 + (gts[matched_indices[pair_id][1], 3] - match_gt_t_map[class_name][frame-1][gt_track_id][1])**2)
              diff_dist = det_dist - gt_dist
              diff_vel_value = diff_value - match_diff_t_map[class_name][frame-1][gt_track_id]
              diff_vel_value[yaw_pos] = angle_in_range(diff_vel_value[yaw_pos])
              diff_vel_value = np.concatenate((diff_vel_value, np.array([diff_dist])))

              diff_vel[class_name].append(diff_vel_value)



  for class_name in KITTI_CLASS_NAMES:
    tmp_diff = np.stack(diff[class_name])
    tmp_dist = np.stack(ego_dist[class_name])
    tmp_angle = np.stack(ego_angle[class_name])
    tmp_crowd = np.stack(gt_crowd[class_name])

    mid_dist_ids = (2 < tmp_dist) & (tmp_dist < 30)
    # plt.scatter(abs(tmp_angle[mid_dist_ids]), abs(tmp_diff[mid_dist_ids, 3]), s=5)
    # plt.scatter(abs(tmp_crowd), abs(tmp_diff[:, 3]), s=5)
    # plt.scatter(tmp_angle[mid_dist_ids], abs(tmp_diff[mid_dist_ids, 5]), s=5)
    # plt.scatter(tmp_angle[mid_dist_ids], abs(tmp_diff[mid_dist_ids, 4]), s=5)
    plt.scatter(tmp_angle[mid_dist_ids], abs(tmp_diff[mid_dist_ids, 3]), s=5)
    # plt.scatter(abs(tmp_diff[:, 1]), abs(tmp_diff[:, 5]))
    # plt.scatter(abs(tmp_dist[mid_dist_ids]), abs(tmp_diff[mid_dist_ids, 6]), s=5)
    plt.show()



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

    # m_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    # m_no_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    # a_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    # for class_name in KITTI_CLASS_NAMES:
    #   if not len(Q_var[class_name]):
    #     continue
    #   m_head_Q[class_name] = [Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11],
    #                              Q_var[class_name][15], Q_var[class_name][11]]
    #   m_no_head_Q[class_name] = [Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11],
    #                                 Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11]]
    #   a_Q[class_name] = [Q_var[class_name][11], Q_var[class_name][10], 0.0, 0.0, 0.0, Q_var[class_name][10]]
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


    m_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    m_no_head_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    a_Q = {class_name: [] for class_name in KITTI_CLASS_NAMES}
    for class_name in KITTI_CLASS_NAMES:
      if not len(Q_var[class_name]):
        continue
      m_head_Q[class_name] = [Q_var[class_name][0], Q_var[class_name][1], Q_var[class_name][3],
                                 Q_var[class_name][15], Q_var[class_name][11]]
      m_no_head_Q[class_name] = [Q_var[class_name][0], Q_var[class_name][1], Q_var[class_name][3],
                                    Q_var[class_name][8], Q_var[class_name][9], Q_var[class_name][11]]
      a_Q[class_name] = [Q_var[class_name][3], Q_var[class_name][2], 0.0, 0.0, 0.0, Q_var[class_name][10]]

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

  print('m_head_P = ', m_head_P)
  print('m_head_Q = ', m_head_Q)
  print('m_no_head_P = ', m_no_head_P)
  print('m_no_head_Q = ', m_no_head_Q)
  print('m_R = ', m_R)
  print('a_P = ', a_P)
  print('a_Q = ', a_Q)
  print('a_R = ', a_R)