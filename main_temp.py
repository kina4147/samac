# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from helpers import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from covariance import Covariance
import json
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox 
from nuscenes.eval.detection.data_classes import DetectionBox 
from pyquaternion import Quaternion
from tqdm import tqdm

import splits


def track_nuscenes():
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
  # if 'mini_val' == data_split:
  #   detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_val.json'
  #   data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini'
  #   version = 'v1.0-mini'
  #   output_path = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_mini_val.json'
  #   scene_splits = splits.mini_val
  # elif 'mini_train' == data_split:
  #   detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_train.json'
  #   data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini'
  #   version = 'v1.0-mini'
  #   output_path = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_mini_train.json'
  #   scene_splits = splits.mini_train


  version = 'v1.0-mini'
  data_root = '/media/marco/60348B1F348AF776/nuscene/raw/v1.0-mini'
  train_detection_file = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_train.json'
  val_detection_file = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_val.json'
  train_output_path = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_mini_train.json'
  val_output_path = '/media/marco/60348B1F348AF776/nuscene/detection/megvii_mini_val.json'
  mini_train_scene_splits = splits.mini_train
  mini_val_scene_splits = splits.mini_val

  train_scene_splits = splits.train
  val_scene_splits = splits.val

  nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

  with open(train_detection_file) as f:
    train_data = json.load(f)

  with open(val_detection_file) as f:
    val_data = json.load(f)

  # train_all_results = EvalBoxes.deserialize(train_data['results'], DetectionBox)
  # val_all_results = EvalBoxes.deserialize(val_data['results'], DetectionBox)
  train_meta = train_data['meta']
  val_meta = val_data['meta']

  train_results = {}
  val_results = {}
  for sample in nusc.sample:
    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)
    if scene['name'] in mini_train_scene_splits:
        print(scene['name'])
        if scene['name'] in train_scene_splits:
            print("train_splits")
            train_results[sample['token']] = train_data['results'][sample['token']]
        else:
            print("val_splits")
            train_results[sample['token']] = val_data['results'][sample['token']]

    if scene['name'] in mini_val_scene_splits:
        print(scene['name'])
        if scene['name'] in train_scene_splits:
            print("train_splits")
            val_results[sample['token']] = train_data['results'][sample['token']]
        else:
            print("val_splits")
            val_results[sample['token']] = val_data['results'][sample['token']]

  # finished tracking all scenes, write output data
  train_output_data = {'meta': train_meta, 'results': train_results}
  with open(train_output_path, 'w') as outfile:
    json.dump(train_output_data, outfile)
  val_output_data = {'meta': val_meta, 'results': val_results}
  with open(val_output_path, 'w') as outfile:
    json.dump(val_output_data, outfile)


if __name__ == '__main__':
  if len(sys.argv)!=1:
    print("Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root")
    sys.exit(1)

  track_nuscenes()

