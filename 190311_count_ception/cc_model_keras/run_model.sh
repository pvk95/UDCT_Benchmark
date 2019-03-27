
#!/bin/bash

python cc_model_keras/count_ception_neurons.py -stride 1 -lr 0.0005 -epochs 1000 -exist True -patch_size 64 -folder 27_Mar -gpu 6  -obj 2 -train_model True -batch 5 -file_count count_maps_64_1 >/dev/null 2>&1 &


python cc_model_keras/count_ception_neurons.py -stride 1 -lr 0.0001 -epochs 1000 -exist True -patch_size 64 -folder 27_Mar_2 -gpu 6  -obj 2 -train_model True -batch 5 -file_count count_maps_64_1 >/dev/null 2>&1 &
