#!/bin/bash

#Analysis for dead neurons
#patch-_size 32 and stride 1
#Generate data.
#Transform the ground truth labels into count maps and augment the dataset to 250 smaples.
python model/data_preprocessing.py --patch_size 32 --stride 1 --filename_count count_maps_32_1.h5 --filename_data dataset_32_1.h5

#Start the simulation.
python model/run_model.py --log_file Jun_17.txt --folderName Jun_17/ --n_sim 2 --n_tr 3 --file_count count_maps_32_1.h5 --filename_data dataset_32_1.h5 --obj 1 --patch_size 32 --epochs 5

#Get the results
python model/all_results.py --folderName Jun_17/ --filename_data dataset_32_1.h5 --obj 0 --genCounts 1 --n_sim 2


#Analysis for live neurons
#patch_size 64
#stride 1
#Generate data
#Transform the ground truth labels into count maps and augment the dataset to 250 smaples.
python model/data_preprocessing.py --patch_size 64 --stride 1 --filename_count count_maps_64_1.h5 --filename_data dataset_64_1.h5

#Start the simulation.
python model/run_model.py --log_file Jun_17.txt --folderName Jun_17/ --n_sim 2 --n_tr 3 --file_count count_maps_64_1.h5 --filename_data dataset_64_1.h5 --obj 2 --patch_size 64 --epochs 5 --lr 0.0005

#Get Results
python model/all_results.py --folderName Jun_17/ --filename_data dataset_64_1.h5 --obj 1 --genCounts 1 --n_sim 2




