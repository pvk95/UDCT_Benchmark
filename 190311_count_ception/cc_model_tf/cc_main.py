#####################################
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import h5py
import tensorflow as tf
import cc_model
import data_preprocessing as pre
from os import environ as cuda_environment
import setproctitle
import sys
import argparse
import time
import pickle
import gc

# python cc_model_keras/cc_main.py -stride 1 -lr 0.005 -epochs 500 -patch_size 64
# -folder 26_Mar -gpu 1  -obj 2 -train_model True -batch 5 -file_count count_maps_64_1
#####################################
# Environment setup

time_begin = time.time()
# Setup for the gpu:
parser = argparse.ArgumentParser(description='Count-ception')

parser.add_argument('-seed', type=int, nargs='?', default=0, help='random seed for split and init')
parser.add_argument('-batch', type=int, nargs='?', default=5, help='batch_size')
parser.add_argument('-stride', type=int, nargs='?', default=1, help='The args.stride at the initial layer')
parser.add_argument('-lr', type=float, nargs='?', default=0.005, help='This will set the learning rate ')
parser.add_argument('-epochs', type=int, nargs='?', default=100, help='No. of epochs ')
parser.add_argument('-patch_size', type=int, nargs='?', default=32, help='patch_size')
parser.add_argument('-train_model', type=int, nargs='?', default=0, help='Train the model')
parser.add_argument('-test_model', type=bool, nargs='?', help='Test the model', default=True)
parser.add_argument('-folder', type=str, nargs='?', help='Name of the experiment', default='cc_model')
parser.add_argument('-gpu', type=str, nargs='?', help='GPU number', default='6')
parser.add_argument('-obj', type=int, nargs='?', help='Object 1 or 2', default=1)
parser.add_argument('-file_count', type=str, nargs='?', help='name of count file', default='count_maps_32_1')
parser.add_argument('-filename_data', type=str, nargs='?', help='name of count file', default='dataset.h5')
parser.add_argument('-sz_tr', type=int, nargs='?', help='size of training data', default=100)

args = parser.parse_args()

if not os.path.exists(args.folder + '/target_' + str(args.obj)):
    os.makedirs(args.folder + '/target_' + str(args.obj))
    save_path = args.folder + '/target_' + str(args.obj)
    with open(save_path + '/log.txt', 'w+') as f:
        f.write("Directory " + args.folder + '/target_' + str(args.obj) + " Created ")
else:
    save_path = args.folder + '/target_' + str(args.obj)
    with open(save_path + '/log.txt', 'w+') as f:
        f.write("Directory " + args.folder + '/target_' + str(args.obj) + " already exists")

with open(save_path + '/log.txt', 'a+') as f:
    f.write("\nParser arguments set: ===>")
    f.write("\nseed: " + str(args.seed))
    f.write("\nbatch: " + str(args.batch))
    f.write("\nstride: " + str(args.stride))
    f.write("\nlr: " + str(args.lr))
    f.write("\nepochs: " + str(args.epochs))
    f.write("\npatch_size: " + str(args.patch_size))
    f.write("\ntrain_model: " + str(args.train_model))
    f.write("\ntest_model: " + str(args.test_model))
    f.write("\nfolder: " + str(args.folder))
    f.write("\ngpu: " + str(args.gpu))
    f.write("\nobj: " + str(args.obj))
    f.write("\nfile_count: " + str(args.file_count))
    f.write("\nTraining data size: " + str(args.sz_tr))
    f.write("\n\ntensorflow" + str(tf.__version__))
    f.write("\n====================>")
'''
username = "Karthik"  # Adapt: Change this string to your username
setproctitle.setproctitle(username)
gpu_number = args.gpu  # Adapt: This is the GPU you are using. Possible values: 0 - 7
cuda_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#####################################
# Read the data and initialize some default args

filename_data = args.filename_data
if not os.path.isfile(filename_data):
    pre.data_augment(filename_count=args.file_count,filename_data=filename_data,total_samples=250)

file_data = h5py.File(args.filename_data, 'r')

img_raw = file_data['raw'][()]
count = file_data['count'][()]
flip_rot = file_data['flip_rot'][()]
total_samples = img_raw.shape[0]

framesize = img_raw.shape[1]
framesize_h = framesize_w = framesize
n_channels = img_raw.shape[3]
input_shape = [framesize_w, framesize_h, n_channels]

n_test = 100
idxs_test = np.arange(total_samples)[-n_test:]
idxs_train = np.random.choice(np.arange(total_samples-n_test), size=args.sz_tr, replace=False)

X_train = img_raw[idxs_train, :, :, :]
y_train = count[idxs_train, :, :, :]

def save_hyper_param(idxs_train, idxs_test):
    hyper_param = {}
    hyper_param['patch_size'] = args.patch_size
    hyper_param['stride'] = args.stride
    hyper_param['idxs_train'] = idxs_train
    hyper_param['idxs_test'] = idxs_test
    hyper_param['flip_rot'] = flip_rot
    hyper_param['lr'] = args.lr
    hyper_param['epochs'] = args.epochs

    # Save the hyper parameters in the training folder
    with open(save_path + "/hyper_param.pickle", 'wb') as f:
        pickle.dump(hyper_param, f)

if (args.obj == 1):
    idx = 0
elif (args.obj == 2):
    idx = 1
else:
    f.write("Unknown object")
    sys.exit(1)

with open(save_path + '/log.txt', 'a+') as f:
    f.write("\n====================>")
    f.write("\nNo. of epochs: " + str(args.epochs))
    f.write("\nBatch Size: " + str(args.batch))
    f.write("\nLearning rate: " + str(args.lr))
    f.write("\nTraining separate model")

y_train = y_train[:, :, :, idx][:, :, :, None]
save_folder = args.folder + '/target_' + str(args.obj)

save_hyper_param(idxs_train, idxs_test)

X_test = img_raw[idxs_test, :, :, :]
y_test = count[idxs_test, :, :, idx][:,:,:,None]

if (args.train_model):
    val_idx = 0
    img_valid = X_test[val_idx, :, :, :]
    count_valid = y_test[val_idx, :, :, :]

    cc = cc_model.cc_model(input_shape=input_shape, patch_size=args.patch_size, stride=args.stride, \
                           lr=args.lr, batch_sz=args.batch, epochs=args.epochs,
                           save_folder = save_folder)

    cc.train_model(X_train=X_train, y_train=y_train, img_valid=img_valid, count_valid=count_valid, \
                   X_valid=X_test, y_valid=y_test, save_folder=save_folder)

if (args.test_model):
    with open(save_path + '/log.txt', 'a+') as f:
        f.write("\n====================>")
        f.write("\nLoading test data")
        f.write("\nSetting default training arguments...")
        f.write("\nSetting default graph...")

    save_folder = args.folder + '/target_' + str(args.obj)

    cc_model.predict(X_test=X_test, save_folder=save_folder)

process_time = (time.time() - time_begin) / 3600
with open(save_path + '/log.txt', 'a+') as f:
    f.write("\n\nProcess ran for: {0:.2f} h".format(process_time))
'''