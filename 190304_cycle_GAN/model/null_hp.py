import numpy as np
import argparse
import glob
import pickle
from scipy import stats

def getRunNames(folder):
    runs = glob.glob(root_folder + folder + 'sim_1/run_*')
    for i in range(len(runs)):
        runs[i] = int(runs[i].split('_')[-1])
    runs = np.sort(runs)
    return runs

##############################################
# Analayze the null hypothesis for the two different types of sampling from B

parser = argparse.ArgumentParser('Null Hypothesis testing')
parser.add_argument('--root_folder',help='Root folder where all simulations were saved',\
                    default='//scratch-second/karthikp/')
parser.add_argument('--folder1',help='Name of folder 1',default='Apr_27/')
parser.add_argument('--folder2',help='Name of folder 2',default='May_26/')

args = parser.parse_args()
root_folder = args.root_folder
folder1 = args.folder1
folder2 = args.folder2

runs_1 = getRunNames(folder1)
runs_2 = getRunNames(folder2)

n_sim = 5
n_runs = len(runs_1)
all_p_less_val = []
cut_off = 0.05/(n_runs*n_sim)

for sim in range(1,n_sim+1):
    run_p_less_val = []
    for i in range(n_runs):
        r1 = runs_1[i]
        r2 = runs_2[i+1]

        with open(root_folder + folder1 + 'sim_{}/run_{}/gt_pred.pkl'.format(sim,r1),'rb') as f:
            gt_pred_1 = pickle.load(f)

        with open(root_folder + folder2 + 'sim_{}/run_{}/gt_pred.pkl'.format(sim,r2),'rb') as f:
            gt_pred_2 = pickle.load(f)

        counts_1 = np.sum(gt_pred_1,axis=(1,2))
        counts_2 = np.sum(gt_pred_2,axis=(1,2))

        output = stats.ttest_1samp(counts_1[:,0]-counts_2[:,0],0)
        t_val = output[0]
        p_2_val = output[1]

        if t_val>0:
            p_less_val = 1 - p_2_val/2
        else:
            p_less_val = p_2_val/2

        run_p_less_val.append(p_less_val)
    run_p_less_val = np.array(run_p_less_val)
    all_p_less_val.append(run_p_less_val)

all_p_less_val = np.array(all_p_less_val)

print(np.mean(all_p_less_val<cut_off))