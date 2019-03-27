#####################################
from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import h5py
import tensorflow as tf
import pickle
import cc_model
import cc_model_combine
import data_preprocessing as pre
from os import environ as cuda_environment

from tensorflow.python.client import device_lib
import sys
import argparse

#####################################
#Environment setup

# Setup for the gpu:
parser = argparse.ArgumentParser(description='Count-ception')

parser.add_argument('-seed', type=int, nargs='?',default=0, help='random seed for split and init')
parser.add_argument('-batch', type=int, nargs='?',default=5, help='batch_size')
parser.add_argument('-stride', type=int, nargs='?',default=1, help='The args.stride at the initial layer')
parser.add_argument('-lr', type=float, nargs='?',default=0.005, help='This will set the learning rate ')
parser.add_argument('-epochs', type=int, nargs='?',default=100, help='No. of epochs ')
parser.add_argument('-count_map_exist', type=bool, nargs='?',default=True, help='whether load the data from disk or generate')
parser.add_argument('-patch_size', type=int, nargs='?',default=32, help='patch_size')
parser.add_argument('-train_model', type=bool, nargs='?', default=False,help='Train the model')
parser.add_argument('-test_model', type=bool, nargs='?', help='Test the model',default=True)
parser.add_argument('-folder', type=str, nargs='?', help='Name of the experiment',default='cc_model')
parser.add_argument('-gpu', type=str, nargs='?', help='GPU number',default='6')
parser.add_argument('-obj', type=int, nargs='?', help='Object 1 or 2',default=1)
parser.add_argument('-combine', type=bool, nargs='?', help='Combined model',default=False)

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("tensorflow", tf.__version__)

'''
import setproctitle
username   = "karthikp" # Adapt: Change this string to your username
setproctitle.setproctitle(username + str(" python 2.7"))
'''
gpu_number = args.gpu   # Adapt: This is the GPU you are using. Possible values: 0 - 7
cuda_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_number)


#####################################
#Read the data and initialize some default args

filename_annotate='Neuron_annotated_dataset.h5'
file_annotate=h5py.File(filename_annotate, 'r')

#col=file_annotate['col']['data']
#gray=file_annotate['gray']['data']

if(not args.count_map_exist):
    gt_AMR=file_annotate['gt_AMR']['data']
    gt_SG=file_annotate['gt_SG']['data']
    gt_SI=file_annotate['gt_SI']['data']

img_raw=file_annotate['raw']['data']

#Default arguments
framesize=img_raw.shape[1]
framesize_h = framesize_w = framesize
n_channels = img_raw.shape[3]
input_shape=[framesize_w,framesize_h,n_channels]

patch_size=args.patch_size

base_y=0 # If you want to crop the image
base_x=0
stride=args.stride # Strides of patch(window)
count_map_exist=args.count_map_exist

#####################################
#Pre processing the data

if(not count_map_exist):
    print("Generating count maps ...")
    print("Expert AMR")
    count_map_AMR,_ = pre.getTrainingData(img_raw=img_raw,annotated=gt_AMR)
    print("Expert SG")
    count_map_SG,_ = pre.getTrainingData(img_raw=img_raw,annotated=gt_SG)
    print("Expert SI")
    count_map_SI,_ = pre.getTrainingData(img_raw=img_raw,annotated=gt_SI)

    #Save the data as h5 file
    file_count='count_maps'
    f=h5py.File(file_count,'w')
    f['count_map_AMR']=count_map_AMR
    f['count_map_SG'] = count_map_SG
    f['count_map_SI'] = count_map_SI
    f.close()
    print("Count maps saved as: ", file_count," in ",str(os.getcwd()))
else:
    print("Loading count maps from h5 file ...")
    file_count = h5py.File('count_maps', 'r')
    count_map_AMR = file_count['count_map_AMR']
    count_map_SG = file_count['count_map_SG']
    count_map_SI = file_count['count_map_SI']

#####################################
#Define the model
count = count_map_AMR
n_outputs=count.shape[3]
assert (n_outputs==2)

print("No. of epochs: ",args.epochs)
print("Batch Size: ",args.batch)
print("Learning rate: ",args.lr)
print("Object %d"%args.obj)
if(args.combine):
    print("Training combined model")

n_samples=img_raw.shape[0]

#idx_shuffle=np.random.randint(0,n_samples,size=n_samples)
idx_shuffle=np.arange(n_samples)
n_train = int(n_samples*0.9)
idxs_train = np.sort(idx_shuffle[:n_train])
idxs_test = np.sort(idx_shuffle[n_train:])

X_train = img_raw[idxs_train,:,:,:]
y_train = count[idxs_train,:,:,:]

X_test = img_raw[idxs_test,:,:,:]
y_test = count[idxs_test,:,:,:]

hyper_param={}
hyper_param['patch_size']=patch_size
hyper_param['stride']=args.stride
hyper_param['idxs_train']=idxs_train
hyper_param['idxs_test']=idxs_test
hyper_param['lr']=args.lr
hyper_param['epochs']=args.epochs
hyper_param['combine']=args.combine

#Save the hyper parameters in the training folder
if not os.path.exists(args.folder):
    os.makedirs(args.folder)
    print("Directory ", args.folder, " Created ")
else:
    print("Directory ", args.folder, " already exists")

pickle_out = open(args.folder+"/hyper_param.pickle",'wb')
pickle.dump(hyper_param,pickle_out)
pickle_out.close()

if(args.combine):
    cc=cc_model_combine.cc_model(input_shape=input_shape,patch_size=patch_size,lr=args.lr,batch_sz=args.batch,epochs=args.epochs)
    model=cc.getModel()

    if(args.train_model):
        cc.train_model(X_train=X_train,y_train=y_train,save_folder = args.folder)

    if(args.test_model):
        cc.predict(X_test=X_test,save_folder = args.folder)

else:
    cc = cc_model.cc_model(input_shape=input_shape, patch_size=patch_size, lr=args.lr, batch_sz=args.batch,
                           epochs=args.epochs)
    model = cc.getModel()

    if (args.train_model):
        if (args.obj == 1):
            idx = 0
        elif (args.obj == 2):
            idx = 1
        else:
            print("Unknown object")
            sys.exit(1)
        y_train_obj = y_train[:, :, :, idx][:, :, :, None]
        cc.train_model(X_train=X_train, y_train=y_train_obj, save_folder=args.folder + '/target_' + str(args.obj))

    if (args.test_model):
        cc.predict(X_test=X_test, save_folder=args.folder + '/target_' + str(args.obj))


file_annotate.close()


'''
with tf.device('/device:GPU:6'):
    cc1=cc_model.cc_model(input_shape=input_shape,patch_size=patch_size,lr=args.lr,batch_sz=args.batch,epochs=args.epochs)
    model1=cc1.getModel()

    if(args.train_model):
        y_train_1 = y_train[:,:,:,0][:,:,:,None]
        cc1.train_model(X_train=X_train,y_train=y_train_1,save_folder = args.folder+'/target_1')

    if(args.test_model):
        y_pred_1=cc1.predict(X_test=X_test,save_folder = args.folder+'/target_1')

with tf.device('/device:GPU:7'):
    cc2=cc_model.cc_model(input_shape=input_shape,patch_size=patch_size,lr=args.lr,batch_sz=args.batch,epochs=args.epochs)
    model2=cc2.getModel()

    if(args.train_model):
        y_train_2 = y_train[:, :, :, 1][:, :, :, None]
        cc2.train_model(X_train=X_train,y_train=y_train_2,save_folder = args.folder+'/target_2')

    if(args.test_model):
        y_pred_2=cc2.predict(X_test=X_test,save_folder = args.folder+'/target_2')

'''