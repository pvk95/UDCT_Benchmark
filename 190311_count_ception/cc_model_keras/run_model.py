'''
This script parallelizes the computational experiments on different GPUs without manual labour of
many python commands.

All the hyperparameters are set constant. Only the dataset sizes are varied.
Data set sizes varied from [10, 20, 30, .....,200]

Caution: Initially for dataset sizes above 100, data augmentation is used.

Caution: First 14 images of the original dataset was left out as test dataset.
Caution: Change -  Last 100 images are test dataset and the remaining are training data. See idxs_train and idxs_test for
any ambiguity
'''

import os
import subprocess
import threading
import time
from datetime import datetime
import numpy as np

total_begin = time.time()

with open("cc_log_2.txt",'w+') as f:
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
        #print("Starting thread {} at : {} ".format(self.name,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        with open("cc_log_2.txt",'a+') as f:
            f.write(print_fn)
        time_begin = time.time()
        process = subprocess.call(self.command)
        proc_dur = (time.time() - time_begin)/3600
        print_fn = "\nExiting thread id:{} name: {} at :{}\n".format(self.threadId,self.name,datetime.now().strftime("%d-%m %H:%M:%S"))
        print_fn = print_fn + "Process took: {} h\n\n".format(proc_dur)
        with open("cc_log_2.txt",'a+') as f:
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

n_sim =10
n_tr = 20 # No. of training dataset sizes
#total_iter = 150*500
#batch_sz = 5
sizes_tr = np.unique(np.around(np.logspace(0,2.17,n_tr))).astype(np.int)
#epochs_szs = (total_iter/sizes_tr).astype(np.int)
#epochs_szs[epochs_szs>total_iter/batch_sz] = total_iter/batch_sz
n_tr = sizes_tr.shape[0]
epochs_szs = [500]*n_tr
n_threads=5 # No. of GPUs to be used. Each thread uses one GPU.

sizes_tr = getOrder(sizes_tr,n_threads)
save_folder = '//scratch-second/karthikp/Apr_7/'

base_args = ["python","cc_model_keras/cc_main.py","-train_model","True","-lr","0.01","-filename_data","dataset_32_1.h5"]

for sim in range(n_sim):
    with open("cc_log_2.txt", 'a+') as f:
        f.write("\n\n=====================================")
        f.write("\nSimulation number: {}\n".format(sim+1))

    runs =[]
    for i in range(n_tr):
        run_name = sizes_tr[i]
        path_folder = save_folder + 'sim_{}/run_{}'.format(sim+1,run_name)
        add = ["-folder",path_folder,"-sz_tr",str(sizes_tr[i]),"-epochs",str(epochs_szs[i])]
        runs.append(base_args + add)

    threads = []

    gpu_threads ={1:3,2:4,3:5,4:6,5:7}

    for i in range(n_threads):
        run_name = "Run_{}".format(sizes_tr[i])
        threads.append(myThread(i+1,run_name,runs[i]+["-gpu",str(gpu_threads[i+1])]))

    for i in range(n_threads):
        threads[i].start()

    run_num = n_threads +1
    while True:
        if(run_num==n_tr+1):
            break
        for i in range(n_threads):
            if threads[i].isAlive():
                continue
            else:
                run_name = "Run_{}".format(sizes_tr[run_num-1])
                threads[i] = myThread(run_num,run_name,runs[run_num-1]+["-gpu",str(gpu_threads[i+1])])
                threads[i].start()
                run_num =run_num+1

    for i in range(n_threads):
        threads[i].join()

total_time = (time.time() - total_begin)/3600
with open("cc_log_2.txt",'a+') as f:
    print_fn = "\n\nEnding simulation at : {} \n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
    f.write("Total time taken: {} h".format(total_time))