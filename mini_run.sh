python main.py mini_val 2 mahal 11 greedy true nuscenes results/0003/test;
python evaluate.py --output_dir results/0003/test --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_val --result_path results/0003/test/mini_val/results.json > results/0003/test/mini_val/output.txt
