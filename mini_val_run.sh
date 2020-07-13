python main.py mini_val 2 iou 0.1 greedy true nuscenes results/0003/iou;
python evaluate.py --output_dir results/0003/iou --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_val --result_path results/0003/iou/mini_val/results.json > results/0003/iou/mini_val/output.txt

python main.py mini_val 2 mahal 11 greedy true nuscenes results/0003/mahal;
python evaluate.py --output_dir results/0003/mahal --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_val --result_path results/0003/mahal/mini_val/results.json > results/0003/mahal/mini_val/output.txt

python main.py mini_val 2 mahal_small 7.5 greedy true nuscenes results/0003/mahal_small;
python evaluate.py --output_dir results/0003/mahal_small --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_val --result_path results/0003/mahal_small/mini_val/results.json > results/0003/mahal_small/mini_val/output.txt

