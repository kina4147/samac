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
from tqdm import tqdm

# from pyquaternion import Quaternion
# from scipy.stats import multivariate_normal as smn
# from filterpy.stats import logpdf #
# from filterpy.stats import multivariate_normal as fmn

from collections import namedtuple
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils.data_classes import Box
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
# AB3DMOT
# import matplotlib; matplotlib.use('Agg')
# from AB3DMOT_libs.model import AB3DMOT
# from AB3DMOT_libs.kitti_utils import Calibration
import os, numpy as np, time, sys
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing
from SAMAC3DMOT.samac import SAMAC3DMOT
from SAMAC3DMOT.utils import angle_in_range, quaternion_to_euler
from pykitti.utils import load_oxts_packets_and_poses, load_velo_scan
from pyquaternion import Quaternion

def track_nuscenes(tracking_name, data_split, covariance_id, association_metric, association_method, save_root):

    save_dir = os.path.join(save_root, data_split);
    mkdir_if_missing(save_dir)
    # scene_splits = None
    # version = None
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

    if tracking_name == 'all':
        NUSCENES_TRACKING_NAMES = ['bicycle', 'bus','car','motorcycle','pedestrian','truck','trailer']
    else:
        NUSCENES_TRACKING_NAMES = [tracking_name]

    point_on = False
    viz_on = True

    results = {}

    total_time, total_frames = 0.0, 0
    with open(detection_file) as f:
        data = json.load(f)

    all_detections = EvalBoxes.deserialize(data['results'], DetectionBox)
    meta = data['meta']

    processed_scene_tokens = set()
    if viz_on:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)

    # multi type tracking
    samac2dataset = [0, 1, 3, 2, 4, 5, 6]
    dataset2samac = [0, 1, 3, 2, 4, 5, 6]
    for sample_token_idx in tqdm(range(len(all_detections.sample_tokens))):
        sample_token = all_detections.sample_tokens[sample_token_idx]
        scene_token = nusc.get('sample', sample_token)['scene_token']
        if scene_token in processed_scene_tokens:
            continue
        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_sample_token = first_sample_token

        # initialize tracker
        mot_trackers = {
            tracking_name: SAMAC3DMOT(covariance_id, dataset2samac=dataset2samac, samac2dataset=samac2dataset, tracking_name=tracking_name) for tracking_name in NUSCENES_TRACKING_NAMES}

        while current_sample_token != '':
            results[current_sample_token] = []
            dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            dboxes = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

            lidar_sd_token = nusc.get('sample', current_sample_token)["data"]["LIDAR_TOP"]
            sd_record = nusc.get("sample_data", lidar_sd_token)
            cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
            if point_on:
                pts = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
                # LIDAR
                lidar_data_path = nusc.get_sample_data_path(lidar_sd_token)
                lidar_points = LidarPointCloud.from_file(lidar_data_path)
                # POSE
                # sample_rec = nusc.get('sample', sd_record['sample_token'])
                # lidar_points, times = LidarPointCloud.from_file_multisweep(nusc,
                #                                                  sample_rec,
                #                                                  sd_record['channel'],
                #                                                  'LIDAR_TOP',
                #                                                  nsweeps=5)

            for dbox in all_detections.boxes[current_sample_token]:
                if dbox.detection_name not in NUSCENES_TRACKING_NAMES:
                    continue

                # x, y, z, yaw, w, l, h
                yaw = quaternion_yaw(Quaternion(dbox.rotation))
                yaw = angle_in_range(yaw)
                det = np.array([dbox.translation[0], dbox.translation[1], dbox.translation[2], yaw, dbox.size[0], dbox.size[1], dbox.size[2]])

                # visualization and more info
                box = Box(dbox.translation, dbox.size, Quaternion(dbox.rotation))
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)
                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)
                if point_on:
                    mask = points_in_box(box, lidar_points.points[0:3, :], wlh_factor=1.0)
                    pts[dbox.detection_name].append(lidar_points.points[0:3, mask])
                    dboxes[dbox.detection_name].append(box)

                # more info
                dist = np.sqrt(box.center[0]**2 + box.center[1]**2)
                loc_pos = np.arctan2(box.center[1], box.center[0])
                loc_yaw = quaternion_yaw(box.orientation)
                wscore = 0.5*(1.0 + np.cos(2.0*(loc_yaw - loc_pos)))

                dets[dbox.detection_name].append(det) # detection state
                info[dbox.detection_name].append(np.array([dbox.detection_score, dist, wscore])) # detection score

            dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])} for tracking_name in NUSCENES_TRACKING_NAMES}
            total_frames += 1

            tboxes = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            tpriors = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            tids = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            for tracking_name in NUSCENES_TRACKING_NAMES:
                start_time = time.time()
                trackers, priors = mot_trackers[tracking_name].update(dets_all=dets_all[tracking_name])
                cycle_time = time.time() - start_time
                total_time += cycle_time
                # x, y, z, theta, w, l, h
                for track in trackers:
                    rotation = Quaternion(axis=[0, 0, 1], angle=track[3]).elements
                    result = {
                        'sample_token': current_sample_token,
                        'translation': [track[0], track[1], track[2]],
                        'size': [track[4], track[5], track[6]],
                        'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
                        'velocity': [0, 0],
                        'tracking_id': str(int(track[7])),
                        'tracking_name': tracking_name,
                        'tracking_score': track[8]
                    }
                    results[current_sample_token].append(result)
                    if viz_on:
                        tbox = Box((track[0], track[1], track[2]), (track[4], track[5], track[6]), Quaternion(axis=[0., 0., 1.], radians=track[3]))
                        tboxes[tracking_name].append(tbox)
                tpriors[tracking_name] = priors
                tids[tracking_name] = trackers[:, 7]
            # visualization
            if viz_on:
                # ax.plot(0, 0, '+', color='black')
                for box, info in zip(dboxes['car'], info['car']):
                    # box.translate(-np.array(pose_record['translation']))
                    # box.rotate(Quaternion(pose_record['rotation']).inverse)
                    # box.translate(-np.array(cs_record['translation']))
                    # box.rotate(Quaternion(cs_record['rotation']).inverse)
                    box.wlh = box.wlh * 1.2
                    box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=1)
            #         # ax.text(box.center[0], box.center[1], "D({:.2f}, {:.2f}, {:.2f})".format(info[0], info[1], info[2]), fontsize=10)
            #
            # #     # for ttracking_name in tboxes:
                bar_l = 5.0
                for box, prior, id in zip(tboxes['car'], tpriors['car'], tids['car']):
                    box.translate(-np.array(pose_record['translation']))
                    box.rotate(Quaternion(pose_record['rotation']).inverse)
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    box.wlh = box.wlh * 1.2
                    # fixed_length (2.0) offset_y
                    offset_y = box.wlh[1]
                    bar_x = box.center[0] - bar_l / 2.0
                    bar_y = box.center[1] + offset_y # considering heading!
                    # tracker should has weight of tracker weight
                    ax.add_patch(
                        patches.Rectangle(
                            (bar_x, bar_y),
                            bar_l, 1,
                            edgecolor='blue', facecolor='blue', fill=True
                        ))
                    ax.add_patch(
                        patches.Rectangle(
                            (bar_x, bar_y),
                            bar_l*prior[0], 1,
                            edgecolor='red', facecolor='red', fill=True
                        ))
                    box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)
                    ax.text(box.center[0], box.center[1], "T({}, {:.2f}, {:.2f})".format(id, prior[0], prior[1]), fontsize=10)

                if point_on:
                    ax.scatter(lidar_points.points[0, :], lidar_points.points[1, :], s=0.05, c=np.array([0.0, 0.0, 1.0]), alpha=0.2)
                    for pt in pts['car']:
                        ax.scatter(pt[0, :], pt[1, :], s=0.3, c=np.array([0.0, 1.0, 0.0]), alpha=1.0)
                ax.add_patch(patches.Rectangle((-1.5, -3.0), 3.0, 6.0, edgecolor='black', facecolor='yellow', fill=True))
                ax.add_patch(patches.Polygon(np.array([[-1.4, 0.0], [0.0, 2.9], [1.4, 0.0]]), color='black', fill=True))
                eval_range = 40
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

# import utm
# from .SAMAC3DMOT.utils import rotx, roty, rotz

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

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
        data['P2']  = np.reshape(data['P2'], [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        # self.V2C = data['Tr_velo_to_cam']
        data['Tr_velo_to_cam'] = np.reshape(data['Tr_velo_to_cam'], [3,4])
        data['Tr_imu_to_velo'] = np.reshape(data['Tr_imu_to_velo'], [3,4])
        data['R0_rect'] = np.reshape(data['R0_rect'], [3,3])
        velo_to_cam = np.eye(4, 4)
        imu_to_velo = np.eye(4, 4)
        cam_to_cnt = np.eye(4, 4)
        velo_to_cam[0:3, 0:4] = data['Tr_velo_to_cam']
        imu_to_velo[0:3, 0:4] = data['Tr_imu_to_velo']
        cam_to_cnt[0:3, 0:3] = data['R0_rect']
        # velo_to_cam = inverse_rigid_trans(data['Tr_velo_to_cam'])
        # imu_to_velo = inverse_rigid_trans(data['Tr_imu_to_velo'])
        # cnt_to_imu = np.dot(np.dot(np.linalg.inv(imu_to_velo), np.linalg.inv(velo_to_cam)), np.linalg.inv(cam_to_cnt))
        imu_to_cam = np.dot(cam_to_cnt, np.dot(velo_to_cam, imu_to_velo))
        qvelo_to_cam = Quaternion(matrix=velo_to_cam[0:3, 0:3])
        qimu_to_velo = Quaternion(matrix=imu_to_velo[0:3, 0:3])
        qcam_to_cnt = Quaternion(matrix=cam_to_cnt)
        qimu_to_cam = qcam_to_cnt * qvelo_to_cam * qimu_to_velo
    return imu_to_cam, qimu_to_cam # , qcam_to_imu


def track_kitti(tracking_name, data_split, covariance_id, assocition_metric, association_method, save_root):

    viz_on = False
    if viz_on:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)

    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    result_sha = 'pointrcnn_' + tracking_name + '_' + data_split
    seq_file_list, num_seq = load_list_from_folder(os.path.join('data/KITTI', result_sha))
    total_time, total_frames = 0.0, 0
    save_dir = os.path.join(save_root, result_sha);
    mkdir_if_missing(save_dir)
    eval_dir = os.path.join(save_dir, 'data');
    mkdir_if_missing(eval_dir)
    seq_count = 0
    samac2dataset = [6, 4, 5, 0, 1, 3, 2] # h, w, l, x, y, z, theta
    dataset2samac = [3, 4, 6, 5, 1, 2, 0] # x, y, theta, z, w, l, h
    for seq_file in seq_file_list:
        # print(seq_file)
        _, seq_name, _ = fileparts(seq_file)
        eval_file = os.path.join(eval_dir, seq_name + '.txt');
        eval_file = open(eval_file, 'w')
        oxts_file = os.path.join('data/KITTI/resources/training/oxts', seq_name + '.txt')
        calib_file = os.path.join('data/KITTI/resources/training/calib', seq_name + '.txt')
        save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name);
        mkdir_if_missing(save_trk_dir)

        mot_tracker = SAMAC3DMOT(covariance_id=covariance_id, tracking_name=tracking_name, samac2dataset=samac2dataset, dataset2samac=dataset2samac)
        seq_dets = np.loadtxt(seq_file, delimiter=',')  # load detections, N x 15
        # seq_oxts = np.loadtxt(oxts_file)
        seq_oxts = load_oxts_packets_and_poses(oxts_files=[oxts_file])
        imu_to_cam, qimu_to_cam = read_calib_file(calib_file)
        cam_to_imu = np.linalg.inv(imu_to_cam)
        qcam_to_imu = qimu_to_cam.inverse
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
                   7:14]  # h, w, l, x, y, z, theta in camera coordinate following KITTI convention
            ldets = dets.copy()
            loc_pose = dets[:, 3:6].copy()
            cam_to_world = np.dot(seq_oxts[frame].T_w_imu, cam_to_imu) # 4x4
            qimu_to_world = Quaternion(matrix=seq_oxts[frame].T_w_imu[0:3, 0:3])
            qworld_to_imu = qimu_to_world.inverse

            # pose ego to world
            if len(dets) > 0:
                w_rot_z = []
                for rot_y in dets[:, 6]:
                    qlyaw = Quaternion(axis=[0, 1, 0], radians=rot_y)
                    qgyaw = qimu_to_world * qcam_to_imu * qlyaw
                    rpy = quaternion_to_euler(qgyaw)
                    w_rot_z.append(rpy[2])
                w_rot_z = np.stack(w_rot_z)
                glo_pose = np.dot(cam_to_world[0:3, 0:3], loc_pose.T) + cam_to_world[0:3,3].reshape(-1, 1)
                dets[:, 3:6] = np.transpose(glo_pose)
                dets[:, 6] = w_rot_z

            dets_all = {'dets': dets, 'info': additional_info}

            start_time = time.time()
            trackers, priors = mot_tracker.update(dets_all=dets_all)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            gtrackers = trackers.copy()
            if len(trackers) > 0:
                # pose world to ego
                glo_pose = trackers[:, 3:6].copy()
                world_to_cam = np.linalg.inv(cam_to_world)
                loc_pose = np.dot(world_to_cam[0:3, 0:3], glo_pose.T) + world_to_cam[0:3, 3].reshape(-1, 1)
                trackers[:, 3:6] = np.transpose(loc_pose)
                c_rot_y = []
                w_rot_z = trackers[:, 6].copy()
                for rot_z in w_rot_z:
                    qgyaw = Quaternion(axis=[0, 0, 1], radians=rot_z)
                    qlyaw = qimu_to_cam * qworld_to_imu * qgyaw
                    rpy = quaternion_to_euler(qlyaw)
                    c_rot_y.append(rpy[1])
                c_rot_y = np.stack(c_rot_y)
                trackers[:, 6] = c_rot_y

            # qcam_to_imu.inverse * qimu_to_world.inverse * Quaternion(axis=(0.0, 0.0, 1.0), radians=)
            # saving results, loop over each tracklet
            for track in trackers:
                bbox3d_tmp = track[0:7]  # h, w, l, x, y, z, theta in camera coordinate
                # to local
                id_tmp = track[7]
                ori_tmp = track[8]
                type_tmp = det_id2str[track[9]]
                bbox2d_tmp_trk = track[10:14]
                conf_tmp = track[14]

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

            # # visualization
            if viz_on:
                for det in dets:
                    ax.scatter(det[3], det[4], color='orange', s=80)
                    ax.add_line(mlines.Line2D([det[3], det[3] + 10*np.cos(det[6])], [det[4], det[4] + 10*np.sin(det[6])], color='orange', lw=2))

                for det in ldets:
                    ax.scatter(det[3], det[5], color='orange', s=80)
                    ax.add_line(mlines.Line2D([det[3], det[3] + 10*np.cos(det[6])], [det[5], det[5] - 10*np.sin(det[6])], color='orange', lw=2))

                for track in gtrackers:
                    ax.scatter(track[3], track[4], color='black')
                    ax.text(track[3], track[4], "T({})".format(int(track[7])), fontsize=10)
                    ax.add_line(mlines.Line2D([track[3], track[3] + 5*np.cos(track[6])], [track[4], track[4] + 5*np.sin(track[6])], color='black'))

                for track in trackers:
                    ax.scatter(track[3], track[5], color='black')
                    ax.text(track[3], track[5], "T({})".format(int(track[7])), fontsize=10)
                    ax.add_line(mlines.Line2D([track[3], track[3] + 5*np.cos(track[6])], [track[5], track[5] + 5*np.sin(track[6])], color='black'))

                ax.add_patch(
                    patches.Rectangle((-1.5, -3.0), 3.0, 6.0, edgecolor='black', facecolor='yellow', fill=True))
                ax.add_patch(
                    patches.Polygon(np.array([[-1.4, 0.0], [0.0, 2.9], [1.4, 0.0]]), color='black', fill=True))
                eval_range = 150
                axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
                ax.set_xlim(-axes_limit, axes_limit)
                ax.set_ylim(-axes_limit, axes_limit)
                plt.title(seq_name + ': ' + str(frame))
                plt.pause(1e-3)
                ax.clear()


            total_frames += 1
            save_trk_file.close()
        seq_count += 1
        eval_file.close()
    print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))




if __name__ == '__main__':
    # if len(sys.argv) != 9:
    #     print(
    #         "Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root")
    #     sys.exit(1)

    dataset = sys.argv[1]
    covariance_id = int(sys.argv[2])
    tracking_name = sys.argv[3]
    save_root = os.path.join('./' + sys.argv[4])
    data_split = sys.argv[5]
    association_metric = sys.argv[6]
    association_method = sys.argv[7]
    # cov_id, association_score (mdist, dist, iou), association_method (greedy, hungarian, jpda)
    # print(dataset, covariance_id)
    if dataset == 'KITTI':
        print('track kitti')
        track_kitti(tracking_name, data_split, covariance_id, association_metric, association_method, save_root)
    elif dataset == 'NuScenes':
        print('track nuscenes')
        track_nuscenes(tracking_name, data_split, covariance_id, association_metric, association_method, save_root)

