#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_test_data=path_to_carla_rs/test

fastec_dataset_type=Fastec
fastec_root_path_test_data=path_to_fastec_rs/test/

model_dir=../experiments/pretrained_models/

results_dir=path_to_results_dir/

#
cd deep_unroll_net

python inference.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_test_data \
          --log_dir=$model_dir \
          --net_type='netMiddle' \
          --results_dir=$results_dir \
          --model_label=pretrained_carla \
          --compute_metrics 

python inference.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_test_data \
          --log_dir=$model_dir \
          --net_type='netMiddle' \
          --results_dir=$results_dir \
          --model_label=pretrained_fastec \
          --compute_metrics 
          