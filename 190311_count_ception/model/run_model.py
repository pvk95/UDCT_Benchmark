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
import argparse


global av_gpus
av_gpus = [1, 2, 3, 4, 5, 6, 7]
global failed
failed = {}

#Create a custom Threading function
class myThread(threading.Thread):
    def __init__(self,threadId,name,command = None):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.command = command

    def add_command(self,command,cv,curr_gpu):
        self.command = self.command + command
        self.cv = cv
        self.curr_gpu = curr_gpu

    def run(self):
        print_fn = "Starting thread id:{} name: {} at :{} \n\n".format(self.threadId,self.name,datetime.now().strftime("%d-%m %H:%M:%S"))
        with open(log_file,'a+') as f:
            f.write(print_fn)
        time_begin = time.time()
        process = subprocess.run(self.command)
        proc_dur = (time.time() - time_begin)/3600
        print_fn = "\nExiting thread id:{} name: {} at :{}\n".format(self.threadId,self.name,datetime.now().strftime("%d-%m %H:%M:%S"))
        print_fn = print_fn + "Process took: {} h\n\n".format(proc_dur)
        with open(log_file,'a+') as f:
            f.write(print_fn)
        self.produce_gpu(process.returncode)

    def produce_gpu(self,process):
        with self.cv:
            av_gpus.append(self.curr_gpu)
            if process==1:
                failed[self.name] = True
            self.cv.notifyAll()

def getOrder(sizes_tr, n_threads):
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

def start_thread(curr_thread):
    with cv:
        while len(av_gpus) ==0:
            cv.wait()
        curr_gpu = av_gpus.pop()
        curr_thread.add_command(["-gpu",str(curr_gpu)],cv,curr_gpu)
        curr_thread.start()


if __name__ == '__main__':
    total_begin = time.time()

    parser = argparse.ArgumentParser('Parallel execution of experiments')

    parser.add_argument('--log_file',default='cc_log.txt',help='Name of log file to be created')
    parser.add_argument('--root_folder',default='//scratch-second/karthikp/',help='Root directory to save folder')
    parser.add_argument('--folderName',default='Jun_17/',help='Where to save folder')

    parser.add_argument('--n_sim',default=10,type = int,help='Number of simulations')
    parser.add_argument('--n_tr',default=20,type = int,help='Number of data points')

    #Arguments to be passed to cc_main.py
    parser.add_argument('--file_count', type=str, nargs='?', \
                        help='Name of count file containing count maps', default='count_maps_32_1.h5')
    parser.add_argument('--filename_data',default='dataset_32_1.h5')
    parser.add_argument('--epochs',type=int,default=12000)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--obj', type=int, default=1,help='1 for dead neurons and 2 for live neurons')
    parser.add_argument('--patch_size', type=int, nargs='?', default=32, help='patch_size')


    args = parser.parse_args()

    log_file = args.log_file
    root_folder = args.root_folder
    folderName = args.folderName
    n_sim = args.n_sim
    n_tr = args.n_tr  # No. of training dataset sizes

    file_count = args.file_count
    filename_data = args.filename_data
    epochs = args.epochs
    lr = args.lr
    obj = args.obj
    patch_size = args.patch_size

    with open(log_file,'a+') as f:
        print_fn = "Starting process at : {} ======>\n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
        f.write(print_fn)
        f.write("Argument parsers given: \n")
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg,getattr(args,arg)))
        f.write('\n')

    sizes_tr = np.unique(np.around(np.logspace(0,2.17,n_tr))).astype(np.int)
    n_tr = sizes_tr.shape[0]
    save_folder = root_folder + folderName

    n_gpus = len(av_gpus)
    sizes_tr = getOrder(sizes_tr, n_gpus)

    base_args = ["python","model/cc_main.py",\
                 "-train_model",'1',\
                 '-file_count',file_count,\
                 '-patch_size',str(patch_size),\
                 "-lr",str(lr),\
                 "-filename_data",filename_data,\
                 "-epochs",str(epochs),\
                 '-obj',str(obj)]

    runs = []
    threads = []

    run_id = 1
    for sim in range(n_sim):
        for sz in sizes_tr:
            path_folder = save_folder + 'sim_{}/run_{}/'.format(sim+1,sz)
            add = ["-folder",path_folder,"-sz_tr",str(sz)]
            runs.append(base_args + add)
            threads.append(myThread(run_id, 'sim_{}/run_{}'.format(sim+1,sz), runs[-1]))
            run_id = run_id + 1

    n_threads = len(runs)
    assert n_threads == n_sim*n_tr

    with open(log_file, 'a+') as f:
        f.write("\n\n=====================================")
        f.write("\n No. of simulations: {}".format(n_sim))
        f.write("\n No. of data points per sim: {}".format(n_tr))
        f.write("\n Total no. of runs: {}\n".format(n_threads))

    cv = threading.Condition()

    gpu_intervals = np.arange(n_threads)[::n_gpus]
    for i in range(n_threads):
        if np.any(gpu_intervals == i):
            time.sleep(1)
        start_thread(threads[i])

    for i in range(n_threads):
        threads[i].join()

    total_time = (time.time() - total_begin) / 3600

    with open(log_file, 'a+') as f:
        print_fn = "\n\nEnding simulation at : {} \n".format(datetime.now().strftime("%d-%m %H:%M:%S"))

        for k in failed.keys():
            f.write("Nans occured at: {}\n".format(k))

        f.write("\nTotal time taken: {} h\n".format(total_time))