python main.py mini_val 2 mahal_small 7.5 greedy true nuscenes results/0003/mahal_small;
python evaluate.py --output_dir results/0003/mahal_small --dataroot /media/marco/60348B1F348AF776/nuscene/raw/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_val --result_path results/0003/mahal_small/mini_val/results.json > results/0003/mahal_small/mini_val/output.txt

