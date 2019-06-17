#!/bin/bash/

#Start simulation
python model/run_model.py --log_file temp --folderName temp/ --n_sim 2 --epochs 5

#Get Results
#python model/results.py --folderName Jun_16/ --genCounts 1