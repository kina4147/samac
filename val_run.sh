python main.py mini_val 2 mahal 11 greedy true nuscenes results/mahal;
python evaluate.py --output_dir results/mahal --dataroot /media/marco/60348B1F348AF776/nuscene/raw/v1.0-trainval --config_path eval/tracking_nips_2019.json  --version v1.0-trainval --eval_set val --result_path results/mahal/val/results.json > results/mahal/output.txt
