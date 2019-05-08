'''
Script to get the plot of loss over the size of the training data
Example: python results.py
Goes through all the simulations and generates a smooth plot by name gan_loss.png
Analysis for dead neurons at the moment
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
import argparse
from scipy.ndimage.measurements import label
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage.feature import peak_local_max as find_max
from scipy.ndimage import gaussian_filter as gauss
import glob
import cv2

def getCounts():
    gts = [gt_AMR,gt_SG,gt_SI]
    counts = np.zeros((gt_AMR.shape[0],2,3))
    for i,gt in enumerate(gts):
        counts[:,:,i] = np.sum(gt,axis=(1,2))[:,:2]
    return counts

#Visualize
def visualize(idx=0):
    plt.figure(figsize=(10,7))
    plt.subplot(121)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imgs_raw[idx,:,:,0],cmap='gray')
    plt.title('True image')
    plt.subplot(122)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred[idx,:,:,:])
    plt.title('Generated Image')
    plt.show()

def get_location_alive(im):
    gt_pred      = np.zeros((im.shape[0],im.shape[1]))

    over_map     = (np.sum(im,axis=2,keepdims=True)>3./4)*1.
    norm_im      = np.sqrt(np.sum(np.square(im),axis=2,keepdims=True))
    normed_im    = im/(1e-10+norm_im)*over_map

    good_col     = 1. - ((im[:,:,1] + im[:,:,2])>1.)*1.

    groups = []
    labeled, num_elements = label(over_map)
    for i in range(num_elements):
        a = np.argwhere(labeled==(i+1))
        s = np.argwhere(good_col[a[:,0],a[:,1]])[:,0]
        a = a[s,:]
        if a.shape[0] < 10:
            continue # The shape is too small

        x_val = normed_im[a[:,0],a[:,1],0]
        y_val = normed_im[a[:,0],a[:,1],2]/2. + (1-normed_im[a[:,0],a[:,1],1])/2.
        dist  = (np.square(np.expand_dims(x_val,0) - np.expand_dims(x_val,1)) +\
                 np.square(np.expand_dims(y_val,0) - np.expand_dims(y_val,1)) +\
                 np.eye(x_val.shape[0])*10)

        min_dist = np.zeros(dist.shape[0])
        for j in range(min_dist.shape[0]):
            sort = np.sort(dist[:,j])
            min_dist[j] = sort[5]

        perm  = np.argsort(min_dist)
        perm  = perm[:int(perm.shape[0]/3)]
        x_val = x_val[perm]
        y_val = y_val[perm]
        a     = a[perm,:]

        # Pred gaussian mixture model size
        thr = 0
        n_components = min(5,perm.shape[0])
        while thr < 0.25:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(np.stack([x_val,y_val],1))
            m   = gmm.means_
            d   = np.sqrt(np.square(np.expand_dims(m[:,0],0) - np.expand_dims(m[:,0],1)) +\
                          np.square(np.expand_dims(m[:,1],0) - np.expand_dims(m[:,1],1)) +\
                          np.eye(m.shape[0]))
            thr = np.min(d)
            n_components -= 1
        n_components += 1
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(np.stack([x_val,y_val],1))
        means = np.array(gmm.means_)
        pred  = gmm.predict(np.stack([x_val,y_val],axis=1))
        for j in range(n_components):
            loc = (np.mean(a[np.argwhere(pred == j)[:,0],:],axis=0)+0.5).astype(np.uint8)[:2]
            gt_pred[loc[0],loc[1]] = 1
    return gt_pred

def get_location_dead(im):
    gt_pred = np.zeros((im.shape[0], im.shape[1]))

    smoothed = gauss(np.pad(im, 10, 'constant'), sigma=0.5)
    smoothed = smoothed[10:-10, 10:-10, 10:13]
    coordinates = find_max(smoothed[:, :, 0], min_distance=2, threshold_abs=0.5)
    # 3.8

    for j in range(coordinates.shape[0]):
        gt_pred[coordinates[j, 0], coordinates[j, 1]] = 1

    return gt_pred, smoothed

def getLoss(true_counts,pred_counts,obj=0):
    #mean_counts = np.mean(true_counts[:, obj])
    #mean_pred_counts = np.mean(pred_counts[:, obj])
    diff_counts = np.mean(np.abs(true_counts[:,obj] - pred_counts[:,obj]))
    return diff_counts

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName_data', default='../cc/Neuron_annotated_dataset.h5',
                        help='Path of annotated dataset')
    args = parser.parse_args()
    fileName_data = args.fileName_data
    # fileName_data = '../190311_count_ception/Neuron_annotated_dataset.h5'
    # save_folder = sys.argv[1]
    # save_folder = 'run_1585/'

    file_gt = h5py.File(fileName_data, 'r')
    imgs_raw = file_gt['raw/data']

    gt_AMR = file_gt['gt_AMR/data']
    gt_SG = file_gt['gt_SG/data']
    gt_SI = file_gt['gt_SI/data']

    counts = getCounts()
    true_counts = np.median(counts, axis=2)
    n_sim = 5
    all_sim = []

    for sim in range(1,n_sim+1):
        save_folder = '//scratch-second/karthikp/Apr_27/sim_{}/'.format(sim)
        all_diff = []
        runs = []
        where_nans = {}

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
            with open(run_folder + 'log.txt') as f:
                content = f.readlines()

            for cont in content[-10:]:
                if 'nan' in cont.lower():
                    k = 'sim_{}'.format(sim)
                    if k not in where_nans.keys():
                        where_nans[k] = [runs[i]]
                    else:
                        where_nans[k].append(runs[i])
                    break

            if os.path.isfile(run_folder + 'CGANdata_gen_B.h5'):
                file_pred = h5py.File(run_folder + 'CGANdata_gen_B.h5', 'r')
            else:
                os.system('python model/main.py save_folder={} valid_file={} mode=data_gen_B gpu=0'.format(run_folder,fileName_data))
                file_pred = h5py.File(run_folder + 'CGANdata_gen_B.h5','r')

            y_pred = file_pred['B/data']

            if y_pred.dtype == 'uint16':
                y_pred = np.array(y_pred) / (2 ** 16 - 1)

            if os.path.isfile(run_folder + 'gt_pred.h5'):
                file_gt_pred = h5py.File(run_folder + 'gt_pred.h5', 'r')
                gt_pred = file_gt_pred['data'][()]
            else:
                gt_pred = np.zeros_like(gt_AMR)
                for i in range(gt_pred.shape[0]):
                    gt_pred[i, :, :, 0], _ = get_location_dead(y_pred[i, :, :, :])
                    # gt_pred[i,:,:,1] = get_location_alive(y_pred[i,:,:,:])
                file_gt_pred = h5py.File(run_folder + 'gt_pred.h5', 'w')
                file_gt_pred['data'] = gt_pred

            pred_counts = np.sum(gt_pred, axis=(1, 2))
            diff_counts = getLoss(true_counts,pred_counts)
            all_diff.append(diff_counts)

        all_diff = np.array(all_diff)
        all_sim.append(all_diff)

    for k in where_nans.keys():
        print("Nans for {} occured at: {}".format(k, where_nans[k]))

    all_sim = np.array(all_sim)

    assert all_sim.shape[0] == n_sim
    assert all_sim.shape[1] == len(runs)

    print(all_sim)

    mean_loss = np.mean(all_sim,axis=0)

    print(mean_loss.shape)

    plt.figure(figsize=(10,8))
    plt.plot(runs,mean_loss,label = 'Test data')
    plt.xlabel('Size of data')
    plt.ylabel('Loss')
    plt.ylim([0,100])
    plt.xlim([0,1800])
    plt.grid()
    plt.title('Loss over size of training data\nUnsupervised learning')
    plt.legend()
    plt.savefig(save_folder + '../gan_loss.png',dpi = 200)



'''
while True:
    print("Press t to terminate: \n")
    idx = input("Enter an index: ")
    if(idx=='t'):
        break
    idx = int(idx)
    visualize(idx)
'''

'''
idx = 5
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(gt_SI[idx,:,:,:]+imgs_raw[idx,:,:,:]/2.)
    plt.title('ground truth')
    plt.subplot(1,2,2)
    plt.title('predicted')
    plt.imshow(gt_pred[idx,:,:,:]+imgs_raw[idx,:,:,:]/2.)
    plt.show()
'''

'''
import numpy as np

def get_b_order(num_samples, idx_start=0):
    if idx_start >= 1728:
        idx_start = idx_start - 1728
        idx_end = idx_start + num_samples
        if idx_end < 1728:
            b_order = np.arange(idx_start, idx_end)
        else:
            b_order = np.concatenate((np.arange(idx_start, 1728), np.arange(idx_end - 1728)))
            idx_end = idx_end - 1728

    else:
        idx_end = idx_start + num_samples
        if idx_end < 1728:
            b_order = np.arange(idx_start, idx_end)
        else:
            b_order = np.concatenate((np.arange(idx_start, 1728), np.arange(idx_end - 1728)))
            idx_end = idx_end - 1728

    assert len(b_order) == num_samples
    return [b_order, idx_end]

num_samples = 200
idx_start = 0
for curr_epoch in range(50):
    print(idx_start ,'--->')
    [b_order,idx_start] = get_b_order(num_samples,idx_start)
    print(idx_start)
'''
