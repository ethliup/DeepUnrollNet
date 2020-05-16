#!/bin/bash

# create an empty folder for pretrained models 
mkdir -p experiments/pretrained_models

# download pretrained models 
cd experiments/pretrained_models
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=10ThiIUKW3kEWXI_v4ZeS0DD7I_jdJa6m' -O pretrained_carla_net_G.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=18tCt8q9kVGrImgmxaUiCt6JSFi5EwJLZ' -O pretrained_fastec_net_G.pth 

