import subprocess
import threading
import time
from datetime import datetime
import numpy as np
import argparse

global av_gpus
av_gpus = [1,2,3,4,5,6,7]
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

def start_thread(curr_thread):
    with cv:
        while len(av_gpus) ==0:
            cv.wait()
        curr_gpu = av_gpus.pop()
        curr_thread.add_command(["--gpu" , str(curr_gpu)],cv,curr_gpu)
        curr_thread.start()

if __name__ == '__main__':

    total_begin = time.time()
    alpha_values = np.arange(0.5,1.5,0.1)

    parser = argparse.ArgumentParser('Alpha value simulation')
    parser.add_argument('--fileName',default='alpha_sim.txt',type=str,help='Name of log file')
    parser.add_argument('--root_folder',default='//scratch-second/karthikp/',help='Root folder')
    parser.add_argument('--folderName',default='Jun_1/',type=str,help='Root folder to save simulation results')
    parser.add_argument('--n_sim',default=5,type=int,help='No. of simulations')
    parser.add_argument('--n_samples', default=1000, type=int, help='No. of samples for simulation')
    args = parser.parse_args()

    fileName = args.fileName
    folderName = args.folderName
    root_folder = args.root_folder
    save_folder = root_folder + folderName
    n_sim = args.n_sim
    n_samples = args.n_samples
    n_gpus = len(av_gpus)
    run_cmds = []
    threads = []
    run_id = 1

    with open(fileName,'a+') as f:
        print_fn = "Starting process at : {} ======>\n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
        f.write(print_fn)

    for sim in range(1,n_sim+1):
        for i,alpha in enumerate(alpha_values):
            path_folder = save_folder + 'sim_{}/run_{}/'.format(sim,i+1)
            run_cmds.append(['python', 'model/generateData.py',\
                             '--data_path' ,save_folder,\
                             '--data_name','dataset_B_{}.h5'.format(i+1),\
                             '--save_path',path_folder,\
                             '--alpha',str(alpha),\
                             '--n_imgs',str(n_samples)])
            threads.append(myThread(run_id,'sim_{}/run_{}'.format(sim,i),run_cmds[-1]))
            run_id = run_id + 1

    n_threads = len(run_cmds)

    with open(fileName, 'a+') as f:
        f.write("\n\n=====================================")
        f.write("\n No. of simulations: {}".format(n_sim))
        f.write("\n No. of data points per sim: {}".format(len(alpha_values)))
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

    with open(fileName, 'a+') as f:
        print_fn = "\n\nEnding simulation at : {} \n".format(datetime.now().strftime("%d-%m %H:%M:%S"))
        f.write("Total time taken: {} h".format(total_time))



