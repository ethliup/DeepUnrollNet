#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_test_data=/media/peidong/Windows1/2020-05-15-leonhard-cluster-backup/datasets/unreal/2019_10_20_Carla_RS_dataset/test

fastec_dataset_type=Fastec
fastec_root_path_test_data=/media/peidong/Windows1/2020-05-15-leonhard-cluster-backup/datasets/fastec/rolling_shutter/test_0206/

model_dir=../experiments/pretrained_models/

results_dir=/home/peidong/Desktop/deep_unroll_results/

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
          