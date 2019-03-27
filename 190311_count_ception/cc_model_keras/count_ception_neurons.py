#####################################
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import h5py
import tensorflow as tf
import cc_model
import cc_model_combine
import data_preprocessing as pre
from os import environ as cuda_environment
import setproctitle
from tensorflow.python.client import device_lib
import sys
import argparse
import time
import pickle



#python cc_model_keras/count_ception_neurons.py -stride 1 -lr 0.005 -epochs 500 -exist True -patch_size 64
# -folder 26_Mar -gpu 1  -obj 2 -train_model True -batch 5 -file_count count_maps_64_1
#####################################
#Environment setup

time_begin =time.time()
# Setup for the gpu:
parser = argparse.ArgumentParser(description='Count-ception')

parser.add_argument('-seed', type=int, nargs='?',default=0, help='random seed for split and init')
parser.add_argument('-batch', type=int, nargs='?',default=5, help='batch_size')
parser.add_argument('-stride', type=int, nargs='?',default=1, help='The args.stride at the initial layer')
parser.add_argument('-lr', type=float, nargs='?',default=0.005, help='This will set the learning rate ')
parser.add_argument('-epochs', type=int, nargs='?',default=100, help='No. of epochs ')
parser.add_argument('-exist', type=str,default='True', help='whether load the data from disk or generate')
parser.add_argument('-patch_size', type=int, nargs='?',default=32, help='patch_size')
parser.add_argument('-train_model', type=bool, nargs='?', default=False,help='Train the model')
parser.add_argument('-test_model', type=bool, nargs='?', help='Test the model',default=True)
parser.add_argument('-folder', type=str, nargs='?', help='Name of the experiment',default='cc_model')
parser.add_argument('-gpu', type=str, nargs='?', help='GPU number',default='6')
parser.add_argument('-obj', type=int, nargs='?', help='Object 1 or 2',default=1)
parser.add_argument('-combine', type=bool, nargs='?', help='Combined model',default=False)
parser.add_argument('-file_count', type=str, nargs='?', help='name of count file',default='count_maps')
parser.add_argument('-frac_data_augment', type=float, nargs='?', help='fraction of data augmentation',default=0.5)

args = parser.parse_args()

if (args.exist == "True"):
    count_map_exist = True
else:
    count_map_exist = False

if (args.combine):
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
        save_path = args.folder
        with open(save_path + '/log.txt', 'w+') as f:
            f.write("Directory "+args.folder+" Created ")
    else:
        save_path = args.folder
        with open(save_path + '/log.txt', 'w+') as f:
            f.write("Directory "+args.folder+" already exists ")
else:
    if not os.path.exists(args.folder+'/target_'+str(args.obj)):
        os.makedirs(args.folder+'/target_'+str(args.obj))
        save_path = args.folder+'/target_'+str(args.obj)
        with open(save_path + '/log.txt', 'w+') as f:
            f.write("Directory "+args.folder + '/target_' + str(args.obj)+" Created ")
    else:
        save_path = args.folder + '/target_' + str(args.obj)
        with open(save_path + '/log.txt', 'w+') as f:
            f.write("Directory "+args.folder+'/target_'+str(args.obj)+" already exists")

with open(save_path+'/log.txt','a+') as f:
    f.write("\nParser arguments set: ===>")
    f.write("\nseed: "+str(args.seed))
    f.write("\nbatch: "+str(args.batch))
    f.write("\nstride: "+str(args.stride))
    f.write("\nlr: "+str(args.lr))
    f.write("\nepochs: "+str(args.epochs))
    f.write("\nexist: "+str(args.exist))
    f.write("\npatch_size: "+str(args.patch_size))
    f.write("\ntrain_model: "+str(args.train_model))
    f.write("\ntest_model: "+str(args.test_model))
    f.write("\nfolder: "+str(args.folder))
    f.write("\ngpu: "+str(args.gpu))
    f.write("\nobj: "+str(args.obj))
    f.write("\ncombine: "+str(args.combine))
    f.write("\nfile_count: "+str(args.file_count))
    f.write("\nfrac_data_augment: "+str(args.frac_data_augment))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    f.write("\ntensorflow"+str(tf.__version__))
    f.write("\n====================>")

username   = "Karthik" # Adapt: Change this string to your username
setproctitle.setproctitle(username)
gpu_number = args.gpu   # Adapt: This is the GPU you are using. Possible values: 0 - 7
cuda_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

#####################################
#Read the data and initialize some default args

np.random.seed(0)
filename_annotate='Neuron_annotated_dataset.h5'
file_annotate=h5py.File(filename_annotate, 'r')

#col=file_annotate['col']['data']
#gray=file_annotate['gray']['data']

if(not count_map_exist):
    gt_AMR=file_annotate['gt_AMR']['data']
    gt_SG=file_annotate['gt_SG']['data']
    gt_SI=file_annotate['gt_SI']['data']

img_raw=file_annotate['raw']['data']
with open(save_path+'/log.txt','a+') as f:
    f.write("\nImage shape: "+str(img_raw.shape[1:]))

#Default arguments
framesize=img_raw.shape[1]
framesize_h = framesize_w = framesize
n_channels = img_raw.shape[3]
input_shape=[framesize_w,framesize_h,n_channels]

# If you want to crop the image
base_y=0
base_x=0
#####################################
#Pre processing the data

if(count_map_exist):
    with open(save_path + '/log.txt', 'a+') as f:
        f.write("\nLoading count maps from h5 file ...")
        file_count = h5py.File(args.file_count, 'r')
        count_map_AMR = file_count['count_map_AMR']
        count_map_SG = file_count['count_map_SG']
        count_map_SI = file_count['count_map_SI']
        f.write("\nShape of Count map: "+str(count_map_AMR.shape[1:]))
else:
    with open(save_path+'/log.txt','a+') as f:
        f.write("\nGenerating count maps ...")
        f.write("\nExpert AMR")
        pre_model=pre.genData(patch_size=args.patch_size,stride=args.stride,base_x=0,base_y=0)
        count_map_AMR, _=pre_model.getTrainingData(img_raw=img_raw, annotated=gt_AMR)
        f.write("\nExpert SG")
        count_map_SG, _ = pre_model.getTrainingData(img_raw=img_raw, annotated=gt_SG)
        f.write("\nExpert SI")
        count_map_SI, _ = pre_model.getTrainingData(img_raw=img_raw, annotated=gt_SI)

        # Save the data as h5 file
        file_count = args.file_count
        file = h5py.File(file_count, 'w')
        file['count_map_AMR'] = count_map_AMR
        file['count_map_SG'] = count_map_SG
        file['count_map_SI'] = count_map_SI
        file.close()
        f.write("\nShape of Count map: "+str(count_map_AMR.shape[1:]))
        f.write("\nCount maps saved as: "+str(file_count)+" in "+str(os.getcwd()))


#####################################
#Define the model
count = np.median((count_map_AMR,count_map_SG,count_map_SI),axis=0)
n_outputs=count.shape[3]
assert (n_outputs==2)
n_samples=img_raw.shape[0]

#####################################
#Data augmentation

with open(save_path+'/log.txt','a+') as f:
    f.write("\n====================>")
    f.write("\nData augmentation ===>")

fraction_augment = args.frac_data_augment
img_augment = []
count_augment = []
for i in range(np.int(n_samples*fraction_augment)):
    idx=np.random.choice(range(n_samples))
    img_modify = img_raw[idx,:,:,:]
    count_modify = count[idx,:,:,:]
    img_modify,count_modify = pre.genRandom(img=img_modify,countMap=count_modify)
    if(img_modify is not None):
        img_augment.append(img_modify)
        count_augment.append(count_modify)

img_augment = np.stack(img_augment)
count_augment = np.stack(count_augment)
img_raw=np.concatenate((img_raw,img_augment),axis=0)
count = np.concatenate((count,count_augment),axis=0)

with open(save_path+'/log.txt','a+') as f:
    f.write("\nNo. of samples after data augmentation: "+str(img_raw.shape[0]))

del img_augment
del count_augment

def save_hyper_param(idxs_train,idxs_test):
    hyper_param={}
    hyper_param['patch_size']=args.patch_size
    hyper_param['stride']=args.stride
    hyper_param['idxs_train']=idxs_train
    hyper_param['idxs_test']=idxs_test
    hyper_param['lr']=args.lr
    hyper_param['epochs']=args.epochs
    hyper_param['combine']=args.combine

    #Save the hyper parameters in the training folder
    with open(save_path+"/hyper_param.pickle",'wb') as f:
        pickle.dump(hyper_param,f)

if(args.combine):

    if(args.train_model):
        with open(save_path + '/log.txt', 'a+') as f:
            f.write("\n====================>")
            f.write("\nNo. of epochs: "+str(args.epochs))
            f.write("\nBatch Size: "+str(args.batch))
            f.write("\nLearning rate: "+str(args.lr))
            f.write("\nTraining combined model")

        idx_shuffle = np.arange(n_samples)
        np.random.shuffle(idx_shuffle)
        n_test = int(n_samples * 0.06)
        idxs_test = np.sort(idx_shuffle[:n_test])
        idxs_concat = np.arange(img_raw.shape[0])[n_samples:]
        idxs_train = np.sort(np.concatenate((idx_shuffle[n_test:],idxs_concat)))

        X_train = img_raw[idxs_train, :, :, :]
        y_train = count[idxs_train, :, :, :]

        X_test = img_raw[idxs_test, :, :, :]
        y_test = count[idxs_test,:,:,:]

        save_hyper_param(idxs_train,idxs_test)
        cc = cc_model_combine.cc_model(input_shape=input_shape, patch_size=args.patch_size, stride=args.stride, \
                                       lr=args.lr, batch_sz=args.batch, epochs=args.epochs)

        cc.train_model(X_train=X_train,y_train=y_train,img_valid=X_test[0,:,:,:],y_valid=y_test,\
                       count_valid=y_test[0,:,:,:],save_folder = args.folder,X_valid=X_test)

    if(args.test_model):
        with open(save_path + '/log.txt', 'a+') as f:
            f.write("\n====================>")
            f.write("\nArgument parsers overwritten")
            if not os.path.exists(args.folder + "/hyper_param.pickle"):
                f.write("\nFile not found! Exiting .....")
                sys.exit(1)
            with open(args.folder + "/hyper_param.pickle",'rb') as f:
                hyper_param = pickle.load(f,encoding="bytes")

        patch_size = hyper_param['patch_size']
        stride = hyper_param['stride']
        idxs_train = hyper_param['idxs_train']
        idxs_test = hyper_param['idxs_test']
        X_test = img_raw[idxs_test, :, :, :]

        cc_model_combine.predict(X_test=X_test,save_folder = args.folder)
        

else:
    if (args.train_model):
        if (args.obj == 1):
            idx = 0
        elif (args.obj == 2):
            idx = 1
        else:
            f.write("Unknown object")
            sys.exit(1)

        with open(save_path + '/log.txt', 'a+') as f:
            f.write("\n====================>")
            f.write("\nNo. of epochs: "+str(args.epochs))
            f.write("\nBatch Size: "+str(args.batch))
            f.write("\nLearning rate: "+str(args.lr))
            f.write("\nTraining separate model")

        idx_shuffle = np.arange(n_samples)
        np.random.shuffle(idx_shuffle)
        n_test = int(img_raw.shape[0] * 0.06)
        idxs_test = np.sort(idx_shuffle[:n_test])
        idxs_concat = np.arange(img_raw.shape[0])[n_samples:]
        idxs_train = np.sort(np.concatenate((idx_shuffle[n_test:], idxs_concat)))

        X_train = img_raw[idxs_train, :, :, :]
        y_train = count[idxs_train, :, :, idx][:,:,:,None]

        X_test = img_raw[idxs_test, :, :, :]
        y_test = count[idxs_test, :, :, idx][:,:,:,None]

        save_hyper_param(idxs_train,idxs_test)

        val_idx=0
        img_valid=X_test[val_idx,:,:,:]
        count_valid = y_test[val_idx, :, :, :]

        cc = cc_model.cc_model(input_shape=input_shape, patch_size=args.patch_size, stride=args.stride, \
                               lr=args.lr, batch_sz=args.batch, epochs=args.epochs,save_folder=args.folder + '/target_' + str(args.obj))

        cc.train_model(X_train=X_train, y_train=y_train, img_valid= img_valid,count_valid=count_valid,\
                       X_valid=X_test,y_valid=y_test,save_folder=args.folder + '/target_' + str(args.obj))

    if (args.test_model):
        with open(save_path + '/log.txt', 'a+') as f:
            f.write("\n====================>")
            f.write("\nArgument parsers overwritten")
            if not os.path.exists(args.folder + '/target_'+str(args.obj)+"/hyper_param.pickle"):
                f.write("\nFile not found! Exiting .....")
                sys.exit(1)

        with open(args.folder + '/target_'+str(args.obj)+"/hyper_param.pickle", 'rb') as f:
            hyper_param = pickle.load(f)

        patch_size = hyper_param['patch_size']
        stride = hyper_param['stride']
        idxs_train = hyper_param['idxs_train']
        idxs_test = hyper_param['idxs_test']
        X_test = img_raw[idxs_test, :, :, :]

        cc_model.predict(X_test=X_test, save_folder=args.folder+'/target_'+str(args.obj))


file_annotate.close()
process_time = (time.time()-time_begin)/3600
with open(save_path+'/log.txt','a+') as f:
    f.write("\n\nProcess ran for: {0:.2f} h".format(process_time))