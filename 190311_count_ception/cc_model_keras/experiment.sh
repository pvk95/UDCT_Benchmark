#!/bin/bash

python cc_model_keras/cc_main.py -train_model True -folder Apr_11 -gpu 7 -filename_data dataset_32_1.h5 -sz_tr 150 -epochs 200 -lr 0.005 &


