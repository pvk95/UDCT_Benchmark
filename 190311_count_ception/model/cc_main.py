'''
# python cc_model_keras/cc_main.py -stride 1 -lr 0.005 -epochs 500 -patch_size 64
# -folder 26_Mar -gpu 1  -obj 2 -train_model True -batch 5 -file_count count_maps_64_1


#Strategy for data augmentation:
There are 114 samples in the annotated dataset. So these 114 images were augmented to upto 250 samples.
This implies then the rest 136 images are a dervative of these 114 images.

Out of these the last 100 images formed part of test size and the first 150 part of training data.
Data sizes sz_tr were sampled from these first 150 images.

'''
#####################################
import os
import numpy as np
import h5py
import cc_model
from os import environ as cuda_environment
import sys
import argparse
import time
import pickle

def save_hyper_param(idxs_train, idxs_test):
    hyper_param = {}
    hyper_param['patch_size'] = var_dict['patch_size']
    hyper_param['stride'] = var_dict['stride']
    hyper_param['idxs_train'] = idxs_train
    hyper_param['idxs_test'] = idxs_test
    hyper_param['flip_rot'] = flip_rot
    hyper_param['lr'] = var_dict['lr']
    hyper_param['epochs'] = var_dict['epochs']

    # Save the hyper parameters in the training folder
    with open(save_path + "hyper_param.pickle", 'wb') as f:
        pickle.dump(hyper_param, f)


if __name__ == '__main__':
    time_begin = time.time()

    #####################################
    # Environment setup

    # Setup for the gpu:
    parser = argparse.ArgumentParser(description='Count-ception')

    #String parser args
    parser.add_argument('-folder', type=str, nargs='?', \
                        help='Name of the experiment', default='cc_model/')
    parser.add_argument('-file_count', type=str, nargs='?', \
                        help='Name of count file containing count maps', default='count_maps_32_1.h5')
    parser.add_argument('-filename_data', type=str, nargs='?', \
                        help='Name of datafile containing aygmented images', default='dataset.h5')

    #Ineteger parser args
    parser.add_argument('-obj', type=int, nargs='?', help='1 for dead neurons and 2 for live neurons', default=1)
    parser.add_argument('-patch_size', type=int, nargs='?', default=32, help='patch_size')
    parser.add_argument('-sz_tr', type=int, nargs='?', help='size of training data', default=100)
    parser.add_argument('-sz_ts', type=int, nargs='?', \
                        help='size of test data. Trailing images considered', default=100)
    parser.add_argument('-seed', type=int, nargs='?', default=0, help='random seed for split and init')
    parser.add_argument('-batch', type=int, nargs='?', default=5, help='batch_size')
    parser.add_argument('-stride', type=int, nargs='?', default=1, help='The stride at the initial layer')
    parser.add_argument('-epochs', type=int, nargs='?', default=100, help='No. of epochs ')
    parser.add_argument('-gpu', type=int, nargs='?', help='GPU number', default=7)

    #Float parser args
    parser.add_argument('-lr', type=float, nargs='?', default=0.005, help='This will set the learning rate ')

    #Boolean parser args
    parser.add_argument('-train_model', type=int, nargs='?', default=0, help='Train the model')
    parser.add_argument('-test_model', type=int, nargs='?', help='Test the model', default=1)

    args = parser.parse_args()

    #Store parser arguments in a dictionary
    var_dict = {}
    for arg in vars(args):
        var_dict[str(arg)] = getattr(args,arg)

    var_dict['train_model'] = not not var_dict['train_model']
    var_dict['test_model'] = not not var_dict['test_model']

    save_path = var_dict['folder'] + 'target_{}/'.format(var_dict['obj'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        with open(save_path + 'log.txt', 'a+') as f:
            f.write("Directory {}/target_{} created\n".format(var_dict['folder'],var_dict['obj']))
    else:
        with open(save_path + 'log.txt', 'a+') as f:
            f.write("Directory {}/target_{} exists\n".format(var_dict['folder'],var_dict['obj']))

    with open(save_path + 'log.txt', 'a+') as f:
        f.write("Parser arguments: \n")
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg,getattr(args,arg)))
        f.write("====================>\n")

    gpu_number = var_dict['gpu']  # Adapt: This is the GPU you are using. Possible values: 0 - 7
    cuda_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #####################################
    # Read the data and initialize some default args
    file_data = h5py.File(var_dict['filename_data'], 'r')

    img_raw = file_data['raw'][()]
    count = file_data['count'][()]
    flip_rot = file_data['flip_rot'][()]
    total_samples = img_raw.shape[0]

    framesize = img_raw.shape[1]
    n_channels = img_raw.shape[3]
    framesize_h = framesize_w = framesize
    input_shape = [framesize_w, framesize_h, n_channels]

    n_test = var_dict['sz_ts']
    idxs_test = np.arange(total_samples)[-n_test:]
    idxs_train = np.random.choice(np.arange(total_samples-n_test), size=var_dict['sz_tr'], replace=False)

    X_train = img_raw[idxs_train, :, :, :]
    y_train = count[idxs_train, :, :, :]

    if (var_dict['obj'] == 1):
        idx = 0
    elif (var_dict['obj'] == 2):
        idx = 1
    else:
        with open(save_path + 'log.txt', 'a+') as f:
            f.write(" Given unknown object\n")
            f.write("Exiting ...\n")
            sys.exit(1)

    with open(save_path + 'log.txt', 'a+') as f:
        f.write("\n====================>\n")
        f.write("No. of epochs: " + str(var_dict['epochs']) + '\n')
        f.write("Batch Size: \n" + str(var_dict['batch'])+ '\n')
        f.write("Learning rate: \n" + str(var_dict['lr'])+ '\n')
        f.write("Training separate model \n")

    y_train = y_train[...,idx][...,None]
    save_hyper_param(idxs_train, idxs_test)

    X_test = img_raw[idxs_test, ...]
    y_test = count[idxs_test, :, :, idx][...,None]

    cc = cc_model.cc_model(input_shape=input_shape, \
                           patch_size=var_dict['patch_size'], \
                           stride=var_dict['stride'], \
                           lr=var_dict['lr'], \
                           batch_sz=var_dict['batch'], \
                           epochs=var_dict['epochs'],\
                           save_folder = save_path,\
                           obj = idx)
    if var_dict['train_model']:
        cc.train_model(X_train=X_train, y_train=y_train, \
                       X_valid=X_test, y_valid=y_test, \
                       val_idx=0)

    if var_dict['test_model']:
        with open(save_path + 'log.txt', 'a+') as f:
            f.write("\n====================>\n")
            f.write("Predictions on test data\n")
            f.write("Loading test data\n")
            f.write("Setting default training arguments...\n")
            f.write("Setting default graph...\n")

        cc.getPredictions(X_test=X_test)


    process_time = (time.time() - time_begin) / 3600
    with open(save_path + 'log.txt', 'a+') as f:
        f.write("\n\nProcess ran for: {0:.2f} h".format(process_time))