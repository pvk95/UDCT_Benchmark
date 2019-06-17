'''
Script to the respective plots


Example: python model/results.py --genCounts 1 --folderName Jun_1/
results are saved in --folderName/report_results
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import glob
import pickle
import subprocess
import threading
import time
import pandas as pd
import os

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

        print("Starting thread: {}".format(self.threadId))
        process = subprocess.run(self.command)
        print("\nEnding thread: {}".format(self.threadId))
        self.produce_gpu(process.returncode)

    def produce_gpu(self,process):
        with self.cv:
            av_gpus.append(self.curr_gpu)
            self.cv.notifyAll()
            if process==1:
                failed[self.name] = True

def start_thread(curr_thread):
    with cv:
        while len(av_gpus) == 0:
            cv.wait()
        curr_gpu = av_gpus.pop()
        curr_thread.add_command([str(curr_gpu)], cv, curr_gpu)
        curr_thread.start()

def getCounts(gt_exp):
    counts = np.zeros((gt_exp[0].shape[0],2,3))
    for i,gt in enumerate(gt_exp):
        #counts[:,:,i] = np.sum(gt,axis=(1,2))[:,:2]
        counts[:, :, i] = np.sum(gt[:,10:-10,10:-10,:], axis=(1, 2))[:, :2]
    return counts


def getExpertLoss(counts):
    n_samples = counts.shape[0]

    counts_matrix = np.zeros(shape=(n_samples, 3, 3, 2))
    for i in range(n_samples):
        for row in range(3):
            for col in range(3):
                counts_matrix[i, row, col, 0] = np.abs(counts[i, 0, row] - counts[i, 0, col])
                counts_matrix[i, row, col, 1] = np.abs(counts[i, 1, row] - counts[i, 1, col])

    experts_diff = np.max(counts_matrix, axis=(1, 2))
    experts_diff_loss = np.mean(experts_diff, axis=0)

    return experts_diff,experts_diff_loss

def getLoss(true_counts,pred_counts,experts_diff):

    diff_counts = np.mean(np.abs(true_counts - pred_counts),axis=0)
    diff_counts_rel = np.mean(np.abs(np.abs(true_counts -pred_counts)-experts_diff)/(true_counts+1),axis=0)
    return diff_counts,diff_counts_rel


def getSegComp(im,gen_im,color_gen_im,fig_title):
    plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im, cmap='gray')
    plt.title('Image to segmented')
    plt.subplot(132)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gen_im)
    plt.title('Generated image')
    plt.subplot(133)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(color_gen_im)
    plt.title('Color transformed image')
    plt.savefig(fig_title,dpi = 200)
    plt.close()

def getSegLoc(im,gen_im,pred_im,fig_title):
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(pred_im + im / 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image and cell positions')
    plt.subplot(122)
    plt.imshow(pred_im + gen_im / 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Generated image and cell positions')
    plt.savefig(fig_title,dpi = 200)
    plt.close()

def save_pd_counts(idx,save_title):
    pd_counts = pd.DataFrame([], columns=['Expert AMR', 'Expert SG', 'Expert SI', 'Median', 'Predicted'])
    pd_counts['Expert AMR'] = counts[:, idx, 0]
    pd_counts['Expert SG'] = counts[:, idx, 1]
    pd_counts['Expert SI'] = counts[:, idx, 2]
    pd_counts['Median'] = true_counts[:, idx]
    pd_counts['Predicted'] = pred_counts[:, idx]
    pd_counts['L1_loss'] = np.abs(true_counts[:, idx] - pred_counts[:, idx])
    pd_counts['Disagreement b/w Experts'] = experts_diff[:, idx]
    pd_counts['Rel_loss'] = (np.abs(np.abs(true_counts[:, idx] - pred_counts[:, idx]) - experts_diff[:, idx])) / (
                true_counts[:, idx] + 1)
    pd_counts.index.name = 'Image'
    if idx ==0:
        pd_counts.columns.name = 'Dead Neurons'
    elif idx ==1:
        pd_counts.columns.name = 'Live Neurons'
    pd_counts.to_html(report_results + save_title)

    return pd_counts

    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName_data', default='/home/karthikp/cc/Neuron_annotated_dataset.h5',
                        help='Path of annotated dataset')
    parser.add_argument('--folderName', help='Root folder of simulations',type=str,default = 'May_26/')
    parser.add_argument('--genCounts', help='Should generate counts?', type=int, default=0)
    parser.add_argument('--mode', help='Whether size or alpha? If Size True else False', type=int, default=1)
    parser.add_argument('--root_folder', help='What is the root folder', type=str, default='//scratch-second/karthikp/')
    parser.add_argument('--n_sim', help='What is the root folder', type=int, default=5)
    args = parser.parse_args()

    fileName_data = args.fileName_data
    folderName = args.folderName
    root_folder = args.root_folder
    genCounts = not not args.genCounts
    mode = not not args.mode
    n_sim = args.n_sim

    #fileName_data = '../190311_count_ception/Neuron_annotated_dataset.h5'
    #folderName = 'Jun_1/'
    #genCounts = False

    file_gt = h5py.File(fileName_data, 'r')
    imgs_raw = file_gt['raw/data'][()]

    gt_AMR = file_gt['gt_AMR/data'][()]
    gt_SG = file_gt['gt_SG/data'][()]
    gt_SI = file_gt['gt_SI/data'][()]
    gt_exp = [gt_AMR,gt_SG,gt_SI]

    # Get the respective counts
    counts = getCounts(gt_exp)
    true_counts = np.median(counts, axis=2)

    experts_diff, experts_diff_loss = getExpertLoss(counts)

    if genCounts:
        n_gpus = len(av_gpus)
        all_runs = []

        for sim in range(1, n_sim + 1):
            save_folder = root_folder + '{}sim_{}/'.format(folderName, sim)
            all_runs = all_runs + glob.glob(save_folder + 'run_*/')

        runs = []
        threads = []
        run_id = 1
        for curr_run in all_runs:
            runs.append(['python','model/getCounts.py' ,fileName_data , curr_run])
            threads.append(myThread(run_id, str(run_id), runs[-1]))
            run_id = run_id + 1

        n_threads = len(runs)
        cv = threading.Condition()

        gpu_intervals = np.arange(n_threads)[::n_gpus]
        for i in range(n_threads):
            if np.any(gpu_intervals == i):
                time.sleep(1)
            start_thread(threads[i])

        for i in range(n_threads):
            threads[i].join()

    all_runs = []
    all_sim = []
    all_sim_rel = []

    for sim in range(1,n_sim+1):
        save_folder = root_folder + '{}sim_{}/'.format(folderName,sim)
        all_diff = []
        all_diff_rel = []
        runs = []

        all_runs = glob.glob(save_folder +'run_*/')
        for run_folder in all_runs:
            temp = run_folder.split('/')[-2].split('_')[-1]
            runs.append(int(temp))

        runs = np.array(runs)
        idxs_sort = np.argsort(runs)

        runs = runs[idxs_sort]
        all_runs = np.array(all_runs)[idxs_sort]

        for i,run_folder in enumerate(all_runs):
            print("Calculating sim_{}/run_{}".format(sim,runs[i]))

            file_pred = h5py.File(run_folder + 'CGAN_gen_B.h5', 'r')
            gen_B = file_pred['B/data'][()]

            with open(run_folder + 'gt_pred.pkl', 'rb') as f:
                gt_pred = pickle.load(f)

            #pred_counts = np.sum(gt_pred, axis=(1, 2))[...,:2]
            pred_counts = np.sum(gt_pred[:,10:-10,10:-10,:], axis=(1, 2))[..., :2]
            diff_counts,diff_counts_rel = getLoss(true_counts, pred_counts,experts_diff)
            all_diff.append(diff_counts)
            all_diff_rel.append(diff_counts_rel)

        all_diff = np.array(all_diff)
        all_diff_rel = np.array(all_diff_rel)
        all_sim.append(all_diff)
        all_sim_rel.append(all_diff_rel)

    if not os.path.exists(root_folder + folderName + 'report_results/'):
        os.makedirs(root_folder + folderName + 'report_results/')

    report_results = root_folder + folderName + 'report_results/'
    ##############################################
    #Analysis for the whole simulation. Results saved in folderName/report_results

    for k in failed.keys():
        print("Nans occured at: {}".format(k))

    all_sim = np.array(all_sim)
    all_sim_rel = np.array(all_sim_rel)

    mean_loss = np.mean(all_sim,axis=0)
    mean_loss_rel = np.mean(all_sim_rel,axis=0)

    std_loss = np.std(all_sim,axis=0)
    std_loss_rel = np.std(all_sim_rel,axis=0)


        
    if mode:
        plt.rcParams.update({'font.size': 18})

        plt.figure(figsize=(20, 8))
        plt.errorbar(runs, mean_loss[:, 0], yerr=std_loss[:, 0], color='m', label='Dead neurons', elinewidth=0.8)
        plt.scatter(runs, mean_loss[:, 0])
        plt.plot(runs, [experts_diff_loss[0]] * len(runs), '--r', label='BL Dead neurons')
        plt.xlabel('Size of training data')
        plt.ylabel('Loss')
        plt.xticks(np.arange(0, 2000, 200))
        plt.ylim([0, 80])
        plt.grid()
        plt.title('Loss over size of training data\nUDCT')
        plt.legend()
        plt.savefig(report_results + 'dead_sz_loss.png', dpi=200)

        plt.figure(figsize=(20, 8))
        plt.errorbar(runs, mean_loss[:, 1], yerr=std_loss[:, 1], color='red', label='Live neurons', elinewidth=0.8)
        plt.scatter(runs, mean_loss[:, 1])
        plt.plot(runs, [experts_diff_loss[1]] * len(runs), '--m', label='BL Live neurons')
        plt.xlabel('Size of training data')
        plt.ylabel('Loss')
        plt.xticks(np.arange(0, 2000, 200))
        plt.ylim([0, 20])
        plt.grid()
        plt.title('Loss over size of training data\nUDCT')
        plt.legend()
        plt.savefig(report_results + 'live_sz_loss.png',dpi = 200)

        plt.figure(figsize=(20, 8))
        plt.errorbar(runs, mean_loss_rel[:, 0], yerr=std_loss[:, 1], color='m', label='Dead neurons', elinewidth=0.8)
        plt.scatter(runs, mean_loss[:, 0])
        plt.xlabel('Size of training data')
        plt.ylabel('Relative Loss')
        plt.xticks(np.arange(0, 2000, 200))
        plt.ylim([0, 2])
        plt.grid()
        plt.title('Loss over size of training data\nUDCT')
        plt.legend()
        plt.savefig(report_results + 'dead_sz_rel_loss.png', dpi=200)

        plt.figure(figsize=(20, 8))
        plt.errorbar(runs, mean_loss[:, 1], yerr=std_loss[:, 1], color='red', label='Live neurons', elinewidth=0.8)
        plt.scatter(runs, mean_loss[:, 1])
        plt.xlabel('Size of training data')
        plt.ylabel('Relative Loss')
        plt.xticks(np.arange(0, 2000, 200))
        plt.ylim([0, 2])
        plt.grid()
        plt.title('Loss over size of training data\nUDCT')
        plt.legend()
        plt.savefig(report_results + 'live_sz_rel_loss.png', dpi=200)

    #alpha_values = np.arange(0.5, 1.5, 0.1)
    #getPlot(alpha_values, mean_loss, std_loss, 'Loss over alpha values', 'Alpha value')

    pd_loss = pd.DataFrame(data=[], index=runs)
    pd_loss['Mean Loss (Dead Neurons)'] = mean_loss[:, 0]
    pd_loss['Std (Dead Neurons)'] = std_loss[:, 1]
    pd_loss['Mean Loss (Live Neurons)'] = mean_loss[:, 1]
    pd_loss['Std (Live Neurons)'] = std_loss[:, 1]
    pd_loss.index.name = 'Size of data'
    pd_loss.to_html(report_results + 'pd_loss.html')

    rel_pd_loss = pd.DataFrame(data=[], index=runs)
    rel_pd_loss['Rel Loss (Dead Neurons)'] = mean_loss_rel[:, 0]
    rel_pd_loss['Std (Dead Neurons)'] = std_loss_rel[:, 1]
    rel_pd_loss['Rel Loss (Live Neurons)'] = mean_loss_rel[:, 1]
    pd_loss['Std (Live Neurons)'] = std_loss_rel[:, 1]
    rel_pd_loss.index.name = 'Size of data'
    rel_pd_loss.to_html(report_results + 'rel_pd_loss.html')

    ##############################################
    #Analysis for a sample of simulation (sim_1/run_1000 set as deafult)
    #Sample Images considered (20,50,100)
    sample_imgs = np.random.choice(np.arange(gen_B.shape[0]),replace=False,size=10)
    save_path = root_folder + folderName + 'sim_1/run_1000/'
    sim_alpha = False

    # save_path = '//scratch-second/karthikp/May_26/sim_1/run_1700/'
    # sim_alpha = False

    if sim_alpha:
        file_dataset_B = h5py.File(save_path + 'dataset_B.h5', 'r')
        alpha = np.array(file_dataset_B['B/alpha'])
        dataset_B = np.array(file_dataset_B['B/data'])
        if dataset_B.dtype == 'uint8':
            dataset_B = np.array(dataset_B) / (2 ** 8 - 1)

        print("Alpha considered: ", alpha)
        print("Shape of dataset_B.h5 :", dataset_B.shape)

    file_gen_B = h5py.File(save_path + 'CGAN_gen_B.h5', 'r')
    gen_B = file_gen_B['B/data']
    if gen_B.dtype == 'uint16':
        gen_B = np.array(gen_B) / (2 ** 16 - 1)

    file_color_gen_B = h5py.File(save_path + 'color_model_gen_B.h5', 'r')
    color_model_gen_B = file_color_gen_B['B/data']
    if color_model_gen_B.dtype == 'uint16':
        color_model_gen_B = np.array(color_model_gen_B) / (2 ** 16 - 1)

    print("Shape of CGAN_gen_B.h5 : ", gen_B.shape)
    print("Shape of color_model_gen_B.h5 : ", color_model_gen_B.shape)

    # Generated preddictions
    with open(save_path + 'gt_pred.pkl', 'rb') as f:
        gt_pred = pickle.load(f)

    for idx in sample_imgs:
        getSegComp(imgs_raw[idx, :, :, 0],gen_B[idx, ...],\
                   color_model_gen_B[idx, ...],report_results+'seg_im_'+str(idx))

        getSegLoc(imgs_raw[idx, ...],gen_B[idx, ...],\
                  gt_pred[idx, ...],report_results+'seg_im_loc_'+str(idx))

    pred_counts = np.sum(gt_pred[:, 10:-10, 10:-10, :], axis=(1, 2))[..., :2]

    dead_pd_counts = save_pd_counts(idx=0, save_title='dead_pd_counts.html')
    live_pd_counts = save_pd_counts(idx = 1,save_title='live_pd_counts.html')

    save_params = {}
    save_params['loss'] = pd_loss
    save_params['dead_pd_counts'] = dead_pd_counts
    save_params['live_pd_counts'] = live_pd_counts

    with open(report_results + 'save_params.pkl','wb') as mydict:
        pickle.dump(save_params,mydict)