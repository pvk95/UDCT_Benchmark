import subprocess
import threading
import time
from datetime import datetime
import numpy as np
import argparse
import h5py
import os

total_begin = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--log_file',type = str,default = 'gan_sim')
args = parser.parse_args()

fileName = args.log_file + '.txt'
with open(fileName,'w+') as f:
    print_fn = "Starting process at : {} ======>\n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
    f.write(print_fn)

#Create a custom Threading function
class myThread(threading.Thread):
    def __init__(self,threadId,name,command):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.command = command

    def run(self):
        print_fn = "Starting thread id:{} name: {} at :{} \n".format(self.threadId,self.name,datetime.now().strftime("%d-%m %H:%M:%S"))
        with open(fileName,'a+') as f:
            f.write(print_fn)
        time_begin = time.time()
        process = subprocess.call(self.command)
        proc_dur = (time.time() - time_begin)/3600
        print_fn = "\nExiting thread id:{} name: {} at :{}\n".format(self.threadId,self.name,datetime.now().strftime("%d-%m %H:%M:%S"))
        print_fn = print_fn + "Process took: {} h\n\n".format(proc_dur)
        with open(fileName,'a+') as f:
            f.write(print_fn)
        return process

def getOrder(sizes_tr,n_threads):
    n_prev = 0
    n_after = len(sizes_tr) - 1
    ascend = True
    descend = False
    new_arr = [0] * len(sizes_tr)
    idx = 0
    while n_prev <= n_after:
        for i in range(n_threads):
            if (n_prev > n_after):
                break
            if (ascend):
                new_arr[idx] = sizes_tr[n_prev]
                idx = idx + 1
                n_prev = n_prev + 1
            elif (descend):
                new_arr[idx] = sizes_tr[n_after]
                idx = idx + 1
                n_after = n_after - 1
        ascend = not ascend
        descend = not descend

    return new_arr

def getNewData(newFileName):
    data_file = h5py.File('dataset_neuron.h5','r')
    images_a = data_file['A/data']

    pred_file = h5py.File('../cc/Neuron_annotated_dataset.h5','r')
    #pred_file = h5py.File('../190311_count_ception/Neuron_annotated_dataset.h5', 'r')
    images_ann = pred_file['raw/data'][:50,:,:,:]

    images_ann = (np.minimum(np.maximum(images_ann, 0), 1) * (2 ** 16 - 1)).astype(np.uint16)
    images_new = np.concatenate((images_ann,images_a),axis=0)

    data_file_new = h5py.File(newFileName,'w')
    group = data_file_new.create_group('A')
    group.create_dataset(name='data', data=images_new, dtype=np.uint16)
    group = data_file_new.create_group('B')
    group.create_dataset(name='data', data=data_file['B/data'], dtype=np.uint16)

    data_file_new.close()
    pred_file.close()
    data_file.close()

newFileName = 'new_dataset_neuron.h5'
if not os.path.isfile(newFileName):
    getNewData(newFileName)

n_sim =5
n_tr = 20 # No. of training dataset sizes
sizes_tr = np.unique(np.around(np.logspace(1,3.23,n_tr))).astype(np.int)
n_tr = sizes_tr.shape[0]
n_threads=7 # No. of GPUs to be used. Each thread uses one GPU.

sizes_tr = getOrder(sizes_tr,n_threads)
save_folder = '//scratch-second/karthikp/Apr_19/'


base_args = ["python","model/main.py"]
runs =[]
gpu_threads ={1:1,2:2,3:3,4:4,5:5,6:6,7:7}
for sim in range(n_sim):
    for i in range(n_tr):
        run_name = sizes_tr[i]
        path_folder = save_folder + 'sim_{}/run_{}'.format(sim+1,run_name)
        add = ["save_folder="+path_folder, "num_samples="+str(sizes_tr[i])]
        runs.append(base_args + add)
tot_runs = len(runs)
with open(fileName, 'a+') as f:
    f.write("\n\n=====================================")
    f.write("\n No. of simulations: {}".format(n_sim))
    f.write("\n No. of data points per sim: {}".format(n_tr))
    f.write("\n Total no. of runs: {}\n".format(tot_runs))

# Initialize threads
threads = []
for i in range(n_threads):
    run_name = "Run_{}".format(sizes_tr[i])
    threads.append(myThread(i+1,run_name,runs[i]+["gpu="+str(gpu_threads[i+1])]))

for i in range(n_threads):
    threads[i].start()

run_num = n_threads
while True:
    if(run_num==tot_runs):
        break
    for i in range(n_threads):
        if threads[i].isAlive():
            continue
        else:
            run_name = "Run_{}".format(sizes_tr[run_num])
            threads[i] = myThread(run_num + 1,run_name,runs[run_num]+["gpu="+str(gpu_threads[i+1])])
            threads[i].start()
            run_num =run_num+1

for i in range(n_threads):
    threads[i].join()

total_time = (time.time() - total_begin)/3600
with open(fileName,'a+') as f:
    print_fn = "\n\nEnding simulation at : {} \n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
    f.write("Total time taken: {} h".format(total_time))

'''
import h5py
f = h5py.File('dataset_neuron.h5','r')
x1 = f['A/data']
print(x1.dtype)
print(x1.shape)

file = h5py.File('new_dataset_neuron.h5','r')
x = file['A/data']
print(x.dtype)
print(x.shape)

y = file['B/data']
print(y.dtype)
print(y.shape)

import matplotlib.pyplot as plt
idx = 50
plt.imshow(x1[idx,:,:,0],cmap='gray')
plt.figure()
plt.imshow(x[idx,:,:,0],cmap='gray')
'''