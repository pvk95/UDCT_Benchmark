'''
Wrapper script to call model/main.py for various runs in parallel.

python model/run_model.py --log_file Jun_15 --folderName Jun_15/
The script first initializes the commands of all the runs and runs them in parallel using as many gpus as available as
specified av_gpus.
'''


import subprocess
import threading
import time
from datetime import datetime
import numpy as np
import argparse

global av_gpus
av_gpus = [2,3,4,5,6,7] #Available GPU indexes.
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
        time_begin = time.time()
        process = subprocess.run(self.command)
        proc_dur = (time.time() - time_begin)/3600
        print_fn = "\nExiting thread id:{} name: {} at :{}\n".format(self.threadId,self.name,datetime.now().strftime("%d-%m %H:%M:%S"))
        print_fn = print_fn + "Process took: {} h\n\n".format(proc_dur)
        with open(fileName,'a+') as f:
            f.write(print_fn)
        self.produce_gpu(process.returncode)


    def produce_gpu(self,process):
        with self.cv:
            av_gpus.append(self.curr_gpu)
            if process==1:
                failed[self.name] = True
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

def start_thread(curr_thread):
    with cv:
        while len(av_gpus) ==0:
            cv.wait()
        curr_gpu = av_gpus.pop()
        curr_thread.add_command(["gpu=" + str(curr_gpu)],cv,curr_gpu)
        curr_thread.start()

if __name__=='__main__':

    total_begin = time.time()
    parser = argparse.ArgumentParser('Size of trainig data simulation')
    parser.add_argument('--log_file',type = str,default = 'gan_sim')
    parser.add_argument('--root_folder',type=str,default='//scratch-second/karthikp/')
    parser.add_argument('--folderName', type=str, default='Jun_15/')
    parser.add_argument('--n_sim', type=int, default=5)

    #Arguments to be passed to main.py
    parser.add_argument('--epochs',type=int,default=200,help='No. of epochs')

    args = parser.parse_args()

    log_file = args.log_file
    root_folder = args.root_folder
    folderName = args.folderName
    epochs = args.epochs

    n_sim = args.n_sim
    fileName = log_file + '.txt'

    with open(fileName,'w+') as f:
        print_fn = "Starting process at : {} ======>\n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
        f.write(print_fn)

    #sizes_tr = np.array([10,25,50,75,100,125,150,175,200,350,500,750,1000,1250,1500,1700]) # Used for May_26/
    sizes_tr = np.array([200, 250])
    n_tr = sizes_tr.shape[0]
    save_folder = root_folder + folderName

    n_gpus = len(av_gpus)
    sizes_tr = getOrder(sizes_tr,n_gpus)

    base_args = ["python","model/main.py"]
    runs =[]
    threads = []

    # Initialize threads
    run_id = 1
    for sim in range(n_sim):
        for sz in sizes_tr:
            path_folder = save_folder + 'sim_{}/run_{}'.format(sim+1,sz)
            add = ["save_folder=" + path_folder, "num_samples=" + str(sz),'epoch={}'.format(epochs)]
            runs.append(base_args + add)
            threads.append(myThread(run_id, 'sim_{}/run_{}'.format(sim+1,sz), runs[-1]))
            run_id +=1

    n_threads= len(runs)
    assert n_threads==run_id-1

    with open(fileName, 'a+') as f:
        f.write("\n\n=====================================")
        f.write("\n No. of simulations: {}".format(n_sim))
        f.write("\n No. of data points per sim: {}".format(n_tr))
        f.write("\n Total no. of runs: {}\n".format(n_threads))

    cv = threading.Condition()


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

        for k in failed.keys():
            f.write("Nans occured at: {}\n".format(k))

        f.write("\nTotal time taken: {} h".format(total_time))

'''
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

'''
