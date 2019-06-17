#!/bin/bash

#Dead Neurons: Simulation in Apr_10/ folder

#Simulate the model for dead neurons
python model/run_model.py --log_file temp.txt --folderName temp/ --epochs 1  --n_sim 2 --n_tr 5 --obj 1 --file_count count_maps_32_1.h5 --filename_data dataset_32_1.h5 --patch_size 32
#Plot the results
python model/all_results.py --folderName temp/ --filename_data dataset_32_1.h5 --obj 0 --genCounts 1 --n_sim 2

#Simulate the model for dead neurons
python model/run_model.py --log_file temp.txt --folderName temp/ --epochs 1  --n_sim 2 --n_tr 5 --obj 2 --file_count count_maps_64_1.h5 --filename_data dataset_64_trial.h5 --patch_size 64

python model/all_results.py --folderName temp/ --filename_data dataset_64_trial.h5 --obj 1 --n_sim 2 --genCounts 1


#Get plots to PC (Live Neurons)
#scp -r karthikp@lbbgpu01://scratch-second/karthikp/Apr_1/report_results_2/ Apr_1/

#Get plots to PC (Dead Neurons)
#scp -r karthikp@lbbgpu01://scratch-second/karthikp/Apr_10/report_results_1/ Apr_10/
