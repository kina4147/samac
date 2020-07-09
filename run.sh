python main.py val 0 iou 0.1 h false nuscenes results/000001; 
python evaluate_nuscenes.py --output_dir results/000001 results/000001/val/results_val_probabilistic_tracking.json > results/000001/output.txt

python main.py val 2 iou 0.01 greedy true nuscenes results/000002; 
python evaluate_nuscenes.py --output_dir results/000002 results/000002/val/results_val_probabilistic_tracking.json > results/000002/output.txt

python main.py val 2 iou 0.1 greedy true nuscenes results/000003; 
python evaluate_nuscenes.py --output_dir results/000003 results/000003/val/results_val_probabilistic_tracking.json > results/000003/output.txt

python main.py val 2 iou 0.25 greedy true nuscenes results/000004; 
python evaluate_nuscenes.py --output_dir results/000004 results/000004/val/results_val_probabilistic_tracking.json > results/000004/output.txt

python main.py val 2 m 11 h true nuscenes results/000005; 
python evaluate_nuscenes.py --output_dir results/000005 results/000005/val/results_val_probabilistic_tracking.json > results/000005/output.txt

python main.py val 0 m 11 greedy true nuscenes results/000006; 
python evaluate_nuscenes.py --output_dir results/000006 results/000006/val/results_val_probabilistic_tracking.json > results/000006/output.txt

python main.py val 2 m 11 greedy false nuscenes results/000007; 
python evaluate_nuscenes.py --output_dir results/000007 results/000007/val/results_val_probabilistic_tracking.json > results/000007/output.txt

python main.py val 2 m 11 greedy true nuscenes results/000008; 
python evaluate_nuscenes.py --output_dir results/000008 results/000008/val/results_val_probabilistic_tracking.json > results/000008/output.txt

python evaluate_nuscenes.py --output_dir results/000008 --dataroot /Users/marco/Desktop/My/my_research/xyztracker/dataset/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_val results/000008/mini_val/results_mini_val_probabilistic_tracking.json > results/000008/output.txt

python evaluate_nuscenes.py ~/datmo/src/data/results/nuscene/sm_rm/results.json --output_dir ~/datmo/src/data/eval/nuscenes_metrics --eval_set val --dataroot /media/marco/60348B1F348AF776/nuscene/raw/v1.0-trainval/ --version v1.0-trainval --config_path ./configs/tracking_nips_2019.json


python evaluate_nuscenes.py --output_dir results/000008 --dataroot /media/marco/60348B1F348AF776/nuscene/raw/v1.0-mini/ --config_path eval/tracking_nips_2019.json  --version v1.0-mini --eval_set mini_val --result_path results/000008/mini_val/results_mini_val_probabilistic_tracking.json > results/000008/output.txt