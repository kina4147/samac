# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from utils import load_list_from_folder, fileparts, mkdir_if_missing
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


def track_nuscenes(data_split):
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
  scene_splits = None
  version = None
  if 'mini_val' == data_split:
    detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_val.json'
    data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini'
    version = 'v1.0-mini'
    output_path = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_mini_val.json'
    scene_splits = splits.mini_val
  elif 'mini_train' == data_split:
    detection_file = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_train.json'
    data_root = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini'
    version = 'v1.0-mini'
    output_path = '/Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/detection/megvii_mini_train.json'
    scene_splits = splits.mini_train
  else:
    print('No Dataset Split', data_split)
    assert (False)


  nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

  with open(detection_file) as f:
    data = json.load(f)
  assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
    'See https://www.nuscenes.org/object-detection for more information.'

  all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
  meta = data['meta']
  print('meta: ', meta)
  print("Loaded results from {}. Found detections for {} samples."
    .format(detection_file, len(all_results.sample_tokens)))


  processed_scene_tokens = set()
  results = {}
  for sample in nusc.sample:
    if sample['token'] in all_results.sample_tokens:
      results[sample['token']] = data['results'][sample['token']]

  # finished tracking all scenes, write output data
  output_data = {'meta': meta, 'results': results}
  with open(output_path, 'w') as outfile:
    json.dump(output_data, outfile)


if __name__ == '__main__':
  if len(sys.argv)!=2:
    print("Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root")
    sys.exit(1)

  data_split = sys.argv[1]
  track_nuscenes(data_split)

