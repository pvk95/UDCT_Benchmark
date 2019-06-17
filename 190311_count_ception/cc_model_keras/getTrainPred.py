#####################################
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import h5py
import tensorflow as tf
import pickle
import cc_model
import sys
import matplotlib.pyplot as plt

def getCounts(labels,ef):
    pest=(np.sum(labels,axis=(1,2))/ef).astype(np.int)
    return pest

def summary_data(true_counts,pred_counts,img_raw,count,idx,obj,save_path):

    n_samples=true_counts.shape[0]

    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(n_samples), true_counts, width=0.4, align='center', label='True counts')
    plt.title('Counts for object '+str(obj+1))
    plt.bar(np.arange(n_samples), pred_counts, width=0.4, align='edge', label='Predicted counts')
    plt.legend()
    plt.savefig(save_path + '/count_'+str(obj+1)+'.png')

    #Regression target
    plt.figure()
    plt.Figure(figsize=(18, 9), dpi=160)
    plt.subplot(1, 3, 1)
    plt.imshow(img_raw[idx, :, :,0], cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(count[idx, :, :, obj],cmap='gray')
    plt.title('Target '+str(obj+1))

    plt.subplot(1, 3, 3)
    if y_pred.shape[3]==2:
        plt.imshow(y_pred[idx, :, :, obj], cmap='gray')
    else:
        plt.imshow(y_pred[idx, :, :, 0], cmap='gray')
    plt.title('Predictions '+str(obj+1))
    plt.savefig(save_path+'/comparison_'+str(idx)+'_obj'+str(obj+1)+'.png')

filename_annotate='Neuron_annotated_dataset.h5'
file_annotate=h5py.File(filename_annotate, 'r')
img_raw=file_annotate['raw']['data'][()]

framesize = 256
framesize_h = framesize_w = framesize
n_channels = 1
input_shape=[framesize_w,framesize_h,n_channels]

combine = sys.argv[1]
save_folder = sys.argv[2]
save_folder = '26_Mar_4/target_2'
if(combine=='True'):
    combine=True
elif(combine=='False'):
    combine=False

if(combine):
    if not os.path.exists(save_folder + "/hyper_param.pickle"):
        print("File not found! Exiting .....")
        sys.exit(1)

    pickle_out = open(save_folder + "/hyper_param.pickle", 'rb')
    hyper_param = pickle.load(pickle_out,encoding="bytes")
    pickle_out.close()

    patch_size = hyper_param['patch_size']
    stride = hyper_param['stride']
    idxs_train = hyper_param['idxs_train']
    idxs_test = hyper_param['idxs_test']
    X_ = img_raw[idxs_test, :, :, :]

    file_count = h5py.File('count_maps_64_1', 'r')
    count_map_AMR = file_count['count_map_AMR']
    count_map_SG = file_count['count_map_SG']
    count_map_SI = file_count['count_map_SI']
    count = np.median((count_map_AMR, count_map_SG, count_map_SI), axis=0)
    count = count[idxs_train[:12],:,:,:]

    cc = cc_model.cc_model(input_shape=input_shape, patch_size=patch_size, stride=stride,save_folder=save_folder)

    checkpoint_path = save_folder + '/checkpoint/cc_model.ckpt'

    model = cc.getModel()
    model.load_weights(checkpoint_path)
    y_pred = model.predict(X_)

    ef = (patch_size/stride)**2

    true_counts = getCounts(count,ef)
    pred_counts = getCounts(y_pred,ef)

    summary_data(true_counts[:,0],pred_counts[:,0],img_raw,count,0,0,save_folder+'/train')
    summary_data(true_counts[:,1], pred_counts[:,1], img_raw, count, 0, 1, save_folder + '/train')