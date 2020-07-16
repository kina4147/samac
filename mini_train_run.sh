python main.py mini_train 2 mahal 11 greedy true nuscenes results/test;
python evaluate.py --output_dir results/test --eval_set mini_train --result_path results/test/mini_train/results.json > results/test/mini_train/output.txt
