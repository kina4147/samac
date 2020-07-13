python main.py mini_train 2 iou 0.1 greedy true nuscenes results/mini_train/iou;
python evaluate.py --output_dir results/mini_train/iou --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_train --result_path results/mini_train/iou/results.json > results/mini_train/iou/output.txt

python main.py mini_train 2 mahal 11 greedy true nuscenes results/mini_train/mahal;
python evaluate.py --output_dir results/0003/mahal --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_train --result_path results/mini_train/mahal/results.json > results/mini_train/mahal/output.txt

python main.py mini_train 2 mahal_small 7.5 greedy true nuscenes results/mini_train/mahal_small;
python evaluate.py --output_dir results/mini_train/mahal_small --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_train --result_path results/mini_train/mahal_small/results.json > results/mini_train/mahal_small/output.txt

