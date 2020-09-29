python main_samac.py KITTI -1 Car results/KITTI val mdist test;
python evaluation/evaluate_kitti3dmot.py pointrcnn_Car_val

python main_samac.py KITTI -1 Pedestrian results/KITTI val mdist test;
python evaluation/evaluate_kitti3dmot.py pointrcnn_Pedestrian_val

python main_samac.py KITTI -1 Cyclist results/KITTI val mdist test;
python evaluation/evaluate_kitti3dmot.py pointrcnn_Cyclist_val
