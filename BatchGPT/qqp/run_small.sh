#!/bin/bash
#python run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='qqp' --batch_size=1 --num_vote=1 --early_stop=200 --json_save=True --load_json=True

python run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='qqp' --batch_size=1 --num_vote=1 --early_stop=100 --json_save=True --load_json=True