'''
Caution: Last 100 images are test dataset and the rest are training dataset
Caution: The hyper parameters used to train all the models was assumed to be same
'''
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from glob import glob
import results
import sys

print("Analyzing all results")
print("====================")

save_folder = '//scratch-second/karthikp/Apr_1/'
#save_folder = './Apr_1/'
cmd = "python cc_model_keras/results.py "
n_sim = 10

run_sim = sys.argv[1] #0 or 1
run_sim = int(run_sim)

if (run_sim):
    loss_sim =[]
    loss_sim_train =[]
    ntr_run_sim =[]
    for i in range(n_sim):
        print("Simulation: {}".format(i+1))
        path_run = save_folder+'sim_{}/*/'.format(i+1)
        path_runs_folder = glob(pathname=path_run)

        loss_run =[]
        loss_run_train =[]
        ntr_run =[]
        for run in path_runs_folder:
            print("\nRun: ",run)
            results_run = results.getResults(run)
            loss_run.append(results_run['diff_counts'])
            loss_run_train.append(results_run['diff_counts_train'])
            ntr_run.append(results_run['n_train'])

        if not loss_run:
            continue
        loss_run = np.array(loss_run)
        loss_run_train = np.array(loss_run_train)
        ntr_run = np.array(ntr_run)
        idxs_sort = np.argsort(ntr_run)
        loss_run = loss_run[idxs_sort]
        loss_run_train = loss_run_train[idxs_sort]
        ntr_run = ntr_run[idxs_sort]
        loss_sim.append(loss_run)
        loss_sim_train.append(loss_run_train)
        ntr_run_sim = ntr_run

    loss_sim = np.stack(loss_sim,axis=0)
    loss_sim_train = np.stack(loss_sim_train,axis=0)
    data_loss = np.mean(loss_sim,axis=0)
    data_loss_std = np.std(loss_sim,axis=0)
    data_loss_train = np.mean(loss_sim_train,axis=0)
    data_loss_train_std = np.std(data_loss_train,axis=0)
    sz_tr = ntr_run_sim

    sim_results ={}
    sim_results['loss_sim'] = loss_sim
    sim_results['loss_sim_train'] = loss_sim_train
    sim_results['data_loss'] = data_loss
    sim_results['data_loss_std'] = data_loss_std
    sim_results['data_loss_train'] = data_loss_train
    sim_results['data_loss_train_std'] = data_loss_train_std
    sim_results['sz_tr'] = sz_tr

    if not os.path.exists(save_folder+ 'total_results'):
        os.makedirs(save_folder+'total_results')

    with open(save_folder + 'total_results/sim_results.pk','wb') as f:
        pickle.dump(sim_results,f)
        print("Results saved as pickle file")
        del sim_results


with open(save_folder+ 'total_results/sim_results.pk','rb') as f:
    print("Loading pickle file")
    sim_results=pickle.load(f,encoding="bytes")

loss_sim = sim_results['loss_sim']
loss_sim_train = sim_results['loss_sim_train']
data_loss = sim_results['data_loss']
data_loss_std = sim_results['data_loss_std']
data_loss_train = sim_results['data_loss_train']
data_loss_train_std = sim_results['data_loss_train_std']
sz_tr = sim_results['sz_tr']

plt.figure(figsize=(8,4),dpi=200)
plt.plot(sz_tr,data_loss,label='Diff_count')
plt.xlabel('Size of training data set')
plt.ylabel('Diff_count')
plt.title('Diff in count (Mean)')
plt.grid()
plt.xticks(np.arange(0,101,10))
plt.ylim([0,30])
plt.legend()
plt.savefig(save_folder+'total_results/diff_count_mean.png',dpi = 200)

plt.figure(figsize=(8,4),dpi=200)
plt.plot(sz_tr,data_loss,label='Test data')
plt.plot(sz_tr,data_loss_train,label = 'Train data')
plt.xlabel('Size of training data set')
plt.ylabel('Diff_count')
plt.title('Diff in count (Mean)')
plt.grid()
plt.xticks(np.arange(0,101,10))
plt.ylim([0,30])
plt.legend()
plt.savefig(save_folder+'total_results/train_test_mean.png',dpi = 200)


data_loss = np.median(loss_sim,axis=0)
data_loss_train = np.median(loss_sim_train,axis=0)

plt.figure(figsize=(8,4),dpi=200)
plt.plot(sz_tr,data_loss,label='Diff_count')
plt.xlabel('Size of training data set')
plt.ylabel('Diff_count')
plt.title('Diff in count (Median)')
plt.grid()
plt.xticks(np.arange(0,101,10))
plt.ylim([0,30])
plt.legend()
plt.savefig(save_folder+'total_results/diff_count_median.png',dpi = 200)

plt.figure(figsize=(8,4),dpi=200)
plt.plot(sz_tr,data_loss,label='Test data')
plt.plot(sz_tr,data_loss_train,label = 'Train data')
plt.xlabel('Size of training data set')
plt.ylabel('Diff_count')
plt.title('Diff in count (Median)')
plt.grid()
plt.xticks(np.arange(0,101,10))
plt.ylim([0,30])
plt.legend()
plt.savefig(save_folder+'total_results/train_test_median.png',dpi = 200)