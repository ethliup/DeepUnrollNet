#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_fastec
mkdir -p experiments/results_demo_carla

cd deep_unroll_net

python inference_demo.py \
            --model_label='pretrained_carla' \
            --log_dir=../experiments/pretrained_models \
            --net_type='netMiddle' \
            --results_dir=../experiments/results_demo_carla \
            --data_dir='../demo/Carla'

python inference_demo.py \
            --model_label='pretrained_fastec' \
            --log_dir=../experiments/pretrained_models \
            --net_type='netMiddle' \
            --results_dir=../experiments/results_demo_fastec \
            --data_dir='../demo/Fastec'
