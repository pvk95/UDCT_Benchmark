import subprocess
import threading
import time
from datetime import datetime
import numpy as np
import argparse
import h5py
import os
import pandas as pd

global av_gpus
av_gpus = [2,3,4,5,6,7]
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
        with open(fileName,'a+') as f:
            f.write(print_fn)
        with self.cv:
            idx, col = self.name.split('/')
            threads_status[col][idx] = '1'
            #threads_status.to_csv(fileName,header=None,index=None,mode='a+')
        time_begin = time.time()
        process = subprocess.run(self.command)
        proc_dur = (time.time() - time_begin)/3600
        print_fn = "\nExiting thread id:{} name: {} at :{}\n".format(self.threadId,self.name,datetime.now().strftime("%d-%m %H:%M:%S"))
        print_fn = print_fn + "Process took: {} h\n\n".format(proc_dur)
        with open(fileName,'a+') as f:
            f.write(print_fn)
        self.produce_gpu(process)

    def produce_gpu(self,process):
        with self.cv:
            av_gpus.append(self.curr_gpu)
            self.cv.notifyAll()


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

if __name__=='__main__':

    total_begin = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file',type = str,default = 'gan_sim')
    args = parser.parse_args()

    fileName = args.log_file + '.txt'

    with open(fileName,'w+') as f:
        print_fn = "Starting process at : {} ======>\n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
        f.write(print_fn)

    newFileName = 'new_dataset_neuron.h5'
    if not os.path.isfile(newFileName):
        getNewData(newFileName)

    n_sim =5
    sizes_tr = np.array([25,50,75,100,125,150,175,200,350,500,750,1000,1250,1500,1700])
    n_tr = sizes_tr.shape[0]
    save_folder = '//scratch-second/karthikp/Apr_27/'
    n_gpus = 6
    sizes_tr = getOrder(sizes_tr,n_gpus)
    pd_data = np.reshape(['0'] *n_tr*n_sim,(n_sim,n_tr))
    threads_status = pd.DataFrame(pd_data,columns= ['run_{}'.format(x) for x in sizes_tr],index=['sim_{}'.format(x) for x in np.arange(1,n_sim+1)])

    base_args = ["python","model/main.py"]
    runs =[]
    threads = []
    # Initialize threads
    run_id = 1
    for sim in range(n_sim):
        for sz in sizes_tr:
            path_folder = save_folder + 'sim_{}/run_{}'.format(sim+1,sz)
            add = ["save_folder=" + path_folder, "num_samples=" + str(sz)]
            runs.append(base_args + add)
            threads.append(myThread(run_id, 'sim_{}/run_{}'.format(sim+1,sz), runs[-1]))
            failed['sim_{}/run_{}'.format(sim+1,sz)] = False
            run_id +=1


    tot_runs = len(runs)
    assert tot_runs==run_id-1
    n_threads = tot_runs

    with open(fileName, 'a+') as f:
        f.write("\n\n=====================================")
        f.write("\n No. of simulations: {}".format(n_sim))
        f.write("\n No. of data points per sim: {}".format(n_tr))
        f.write("\n Total no. of runs: {}\n".format(tot_runs))

    cv = threading.Condition()

    def start_thread(curr_thread):
        with cv:
            while len(av_gpus) ==0:
                cv.wait()
            curr_gpu = av_gpus.pop()
            curr_thread.add_command(["gpu=" + str(curr_gpu)],cv,curr_gpu)
            proc_out = curr_thread.start()

    gpu_intervals = np.arange(n_threads)[::n_gpus]
    for i in range(n_threads):
        if np.any(gpu_intervals==i):
            time.sleep(1)
        start_thread(threads[i])

    for i in range(n_threads):
        threads[i].join()

    total_time = (time.time() - total_begin)/3600
    
    with open(fileName,'a+') as f:
        print_fn = "\n\nEnding simulation at : {} \n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
        f.write("Total time taken: {} h".format(total_time))

