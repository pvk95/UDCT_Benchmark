'''
Script to get the position of live and dead neurons on predicted data.
python model/getCounts.py /home/karthikp/cc/Neuron_annotated_dataset.h5 //scratch-second/karthikp/May_26/sim_1/run_10/ 7
Calculates the counts and stores as a pickle //scratch-second/karthikp/May_26/sim_1/run_10/gt_pred.pkl
gt_pred : [n_samples,im_sz,im_sz,3] 1st and 2nd channels correspond to dead and live neurons respectively
'''

import numpy as np
import h5py
import sys
import os
from scipy.ndimage.measurements import label
from sklearn.mixture import GaussianMixture
from skimage.feature import peak_local_max as find_max
from scipy.ndimage import gaussian_filter as gauss
import pickle

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

def getColorPred(run_folder,gen_B, gpu):
    if not os.path.isfile(run_folder + 'color_model_gen_B.h5'):
        # Get the color generated data
        data_gen_B = gen_B[..., 1][:, :, :, None]
        with h5py.File(run_folder + 'col_data_gen_B.h5', 'w') as file_data_col:
            group_A = file_data_col.create_group('A')
            num_channel = data_gen_B.shape[3]
            num_samples = data_gen_B.shape[0]
            group_A.create_dataset(name='num_channel', data=num_channel, dtype=int)
            group_A.create_dataset(name='num_samples', data=num_samples, dtype=int)
            group_A.create_dataset(name='data', data=data_gen_B, dtype=np.uint16)

            group_B = file_data_col.create_group('B')
            num_channel = 3
            num_samples = 0
            group_B.create_dataset(name='num_channel', data=num_channel, dtype=int)
            group_B.create_dataset(name='num_samples', data=num_samples, dtype=int)
            group_B.create_dataset(name='data', data=np.zeros((1, 256, 256, 3)))

        cmd = 'python model/main.py save_folder={} ckpt_path=1 dataset={}col_data_gen_B.h5 '\
        'mode=gen_B name=color_model gpu={}'.format(run_folder,run_folder,gpu)
        os.system(cmd)
        os.remove(run_folder + 'col_data_gen_B.h5')

if __name__ =='__main__':

    fileName_data = sys.argv[1]
    run_folder = sys.argv[2]
    gpu = sys.argv[3]

    if not os.path.isfile(run_folder + 'CGAN_gen_B.h5'):
        os.system('python model/main.py save_folder={} valid_file={} mode=data_gen_B gpu={}'.format(run_folder,\
                                                                                                    fileName_data,gpu))

    file_pred = h5py.File(run_folder + 'CGAN_gen_B.h5', 'r')
    gen_B = file_pred['B/data'][()]
    n_samples = gen_B.shape[0]
    im_sz = gen_B.shape[1]

    getColorPred(run_folder, gen_B, gpu)

    with h5py.File(run_folder + 'color_model_gen_B.h5', 'r') as file_color:
        color_gen_B = file_color['B/data'][()]
        if color_gen_B.dtype == 'uint16':
            color_gen_B = np.array(color_gen_B) / (2 ** 16 - 1)

    if gen_B.dtype == 'uint16':
        gen_B = np.array(gen_B) / (2 ** 16 - 1)

    #TO DO:
    #order = [2,1,0]
    order = [0,1,2]

    #if not os.path.exists(run_folder + 'gt_pred.pkl'):
    if True:
        gt_pred = np.zeros((n_samples,im_sz,im_sz,3))
        for i in range(gt_pred.shape[0]):
            temp_im = gen_B[i,...]
            temp_im = temp_im[:,:,order]
            gt_pred[i, :, :, 0], _ = get_location_dead(temp_im)
            temp_im = color_gen_B[i, ...]
            temp_im = temp_im[:,:,order]
            gt_pred[i, :, :, 1] = get_location_alive(temp_im)
            with open(run_folder + 'gt_pred.pkl', 'wb') as f:
                pickle.dump(gt_pred, f)

