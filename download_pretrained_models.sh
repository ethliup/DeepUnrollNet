#!/bin/bash

# create an empty folder for pretrained models 
mkdir -p experiments/pretrained_models

# download pretrained models 
cd experiments/pretrained_models
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1T8Kkvm-2DT5Yqv76BKwp03Ox7gGb1LA5' -O pretrained_carla_net_G.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1eZX8RCC2aYQXD__w4WjfpyU04j137VpM' -O pretrained_fastec_net_G.pth 

