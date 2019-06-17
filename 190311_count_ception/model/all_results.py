'''
Caution: Last 100 images are test dataset and the rest are training dataset
Caution: The hyper parameters used to train all the models was assumed to be same
'''
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import results
import argparse
import h5py


def getExpertLoss(annotate_data = '/home/karthikp/cc/Neuron_annotated_dataset.h5'):
    file_gt = h5py.File(annotate_data, 'r')

    gt_AMR = file_gt['gt_AMR/data'][()]
    gt_SG = file_gt['gt_SG/data'][()]
    gt_SI = file_gt['gt_SI/data'][()]
    gt_exp = [gt_AMR, gt_SG, gt_SI]

    counts = np.zeros((gt_exp[0].shape[0], 2, 3))
    for i, gt in enumerate(gt_exp):
        counts[:, :, i] = np.sum(gt, axis=(1, 2))[:, :2]

    n_samples = counts.shape[0]

    counts_matrix = np.zeros(shape=(n_samples, 3, 3, 2))
    for i in range(n_samples):
        for row in range(3):
            for col in range(3):
                counts_matrix[i, row, col, 0] = np.abs(counts[i, 0, row] - counts[i, 0, col])
                counts_matrix[i, row, col, 1] = np.abs(counts[i, 1, row] - counts[i, 1, col])

    experts_diff = np.max(counts_matrix, axis=(1, 2))
    experts_diff_loss = np.mean(experts_diff, axis=0)

    return experts_diff, experts_diff_loss


if __name__ == '__main__':

    print("Analyzing all results")
    print("====================")

    parser = argparse.ArgumentParser('Script to analyze the results of all simulations')

    parser.add_argument('--root_folder', default='//scratch-second/karthikp/', help='Root directory to save folder')
    parser.add_argument('--folderName', default='Jun_17/', help='Where to save folder')
    parser.add_argument('--filename_data', help='The augmented image data file', default='dataset_32_1.h5')
    parser.add_argument('--obj',default=0,type=int,help='0 for dead neurons and 1 for live neurons')
    parser.add_argument('--genCounts',type=int,default=0,help='Whether to generate counts for every run')
    parser.add_argument('--n_sim', type=int, default=10, help='No. of simulations')

    args = parser.parse_args()

    root_folder = args.root_folder
    folderName = args.folderName
    filename_data = args.filename_data
    obj = args.obj
    n_sim = args.n_sim
    genCounts = not not args.genCounts

    save_folder = root_folder + folderName
    report_results = save_folder + 'report_results_{}/'.format(obj + 1)


    if (genCounts):
        loss_sim =[]
        loss_sim_train =[]
        ntr_run_sim =[]

        runs = glob.glob(save_folder + 'sim_1/run_*/')
        for i, curr_run in enumerate(runs):
            runs[i] = int(curr_run.split('_')[-1].split('/')[0])
        runs = np.sort(runs)
        for sim in range(1,n_sim+1):
            loss_run =[]
            loss_run_train =[]
            ntr_run =[]
            for curr_run in runs:
                print('Calculating sim_{}/run_{}'.format(sim,curr_run))
                save_path = save_folder + 'sim_{}/run_{}/'.format(sim,curr_run)
                results_run = results.getResults(save_folder=save_path,\
                                                 filename_data=filename_data,\
                                                 obj=obj)
                loss_run.append(results_run['diff_counts'])
                loss_run_train.append(results_run['diff_counts_train'])
                ntr_run.append(results_run['n_train'])

            loss_run = np.array(loss_run)
            loss_run_train = np.array(loss_run_train)
            ntr_run = np.array(ntr_run)

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

        if not os.path.exists(report_results):
            os.makedirs(report_results)

        with open(report_results + 'sim_results.pkl','wb') as f:
            pickle.dump(sim_results,f)
            print("Results saved as pickle file")

    with open(report_results + 'sim_results.pkl','rb') as f:
        print("Loading pickle file")
        sim_results=pickle.load(f)

    loss_sim = sim_results['loss_sim']
    loss_sim_train = sim_results['loss_sim_train']
    data_loss = sim_results['data_loss']
    data_loss_std = sim_results['data_loss_std']
    data_loss_train = sim_results['data_loss_train']
    data_loss_train_std = sim_results['data_loss_train_std']
    sz_tr = sim_results['sz_tr']

    exprets_diff, experts_diff_loss = getExpertLoss()

    ###########################################
    #Mean loss: One figure for just test data while other is for test and training data

    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(20, 8), dpi=200)
    plt.errorbar(sz_tr, data_loss, yerr=data_loss_std, color='m', label='Test loss', elinewidth=0.8)
    plt.scatter(sz_tr, data_loss)
    plt.plot(sz_tr, [experts_diff_loss[obj]] * len(sz_tr), '--b', label='Disagreement b/w experts')
    plt.xlabel('Size of training data set')
    plt.ylabel('Test Loss')
    plt.title('Loss over size of training data\n Count-Ception')
    plt.grid()
    plt.xticks(np.arange(0, 151, 10))
    plt.ylim([0, 50])
    plt.legend()
    plt.savefig(report_results + 'diff_count_mean.png', dpi=200)

    plt.figure(figsize=(20, 8), dpi=200)
    plt.errorbar(sz_tr, data_loss, yerr=data_loss_std, color='m', label='Test Loss')
    plt.errorbar(sz_tr, data_loss_train, yerr=data_loss_train_std, color='red', label='Training Loss')
    plt.plot(sz_tr, [experts_diff_loss[obj]] * len(sz_tr), '--b', label='Disagreement b/w experts')
    plt.scatter(sz_tr, data_loss)
    plt.scatter(sz_tr, data_loss_train)
    plt.xlabel('Size of training data set')
    plt.ylabel('Loss')
    plt.title('Loss over size of training data\n Count-Ception')
    plt.grid()
    plt.xticks(np.arange(0, 151, 10))
    plt.ylim([0, 50])
    plt.legend()
    plt.savefig(report_results + 'train_test_mean.png', dpi=200)

    ###########################################
    # Median loss: One figure for just test data while other is for test and training data

    data_loss = np.median(loss_sim,axis=0)
    data_loss_train = np.median(loss_sim_train,axis=0)

    plt.figure(figsize=(20, 8), dpi=200)
    plt.errorbar(sz_tr, data_loss, yerr=data_loss_std, color='m', label='Test loss', elinewidth=0.8)
    plt.scatter(sz_tr, data_loss)
    plt.plot(sz_tr, [experts_diff_loss[obj]] * len(sz_tr), '--b', label='Disagreement b/w experts')
    plt.xlabel('Size of training data set')
    plt.ylabel('Test Loss')
    plt.title('Loss over size of training data\n Count-Ception')
    plt.grid()
    plt.xticks(np.arange(0, 151, 10))
    plt.ylim([0, 50])
    plt.legend()
    plt.savefig(report_results + 'diff_count_median.png',dpi = 200)

    plt.figure(figsize=(20, 8), dpi=200)
    plt.errorbar(sz_tr, data_loss, yerr=data_loss_std, color='m', label='Test Loss')
    plt.errorbar(sz_tr, data_loss_train, yerr=data_loss_train_std, color='red', label='Training Loss')
    plt.plot(sz_tr, [experts_diff_loss[obj]] * len(sz_tr), '--b', label='Disagreement b/w experts')
    plt.scatter(sz_tr, data_loss)
    plt.scatter(sz_tr, data_loss_train)
    plt.xlabel('Size of training data set')
    plt.ylabel('Loss')
    plt.title('Loss over size of training data\n Count-Ception')
    plt.grid()
    plt.xticks(np.arange(0, 151, 10))
    plt.ylim([0, 50])
    plt.legend()
    plt.savefig(report_results + 'train_test_median.png',dpi = 200)
