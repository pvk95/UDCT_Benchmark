import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
sys.path.append('./model/')
import cv2
import os
import csv
import pickle
import argparse
from all_results import getExpertLoss


experts_diff,experts_diff_loss = getExpertLoss()

def fix_ambiguity():
    filename_annotate = 'Neuron_annotated_dataset.h5'
    file_annotate = h5py.File(filename_annotate, 'r')
    img_raw = file_annotate['raw']['data'][()]

    filename_count = 'count_maps_64_1.h5'
    file_count = h5py.File(filename_count, 'r')
    count_map_AMR = file_count['count_map_AMR'][()]
    count_map_SG = file_count['count_map_SG'][()]
    count_map_SI = file_count['count_map_SI'][()]

    count = np.median((count_map_AMR,count_map_SG,count_map_SI),axis=0)

    with open('//scratch-second/karthikp/Apr_1/sim_1/run_100/target_2/hyper_param.pickle', 'rb') as mydict:
        hyper_param = pickle.load(mydict)

    flip_rot = hyper_param['flip_rot']
    idxs_test = hyper_param['idxs_test']

    # Data augmentation on test data set according flip_rot of hyper_param
    imgs_augment = []
    count_augment = []

    for i in range(flip_rot.shape[0]):
        idxs_aug = flip_rot[i, :]  # idxs_aug[0]: Image, idxs_aug[1]: flip idxs_aug[2]:rot
        im2aug = img_raw[idxs_aug[0], :, :, :]
        count2aug = count[idxs_aug[0], :, :, :]

        if (idxs_aug[1]):
            im2aug = np.flipud(im2aug)
            count2aug = np.flipud(count2aug)

        im2aug = np.rot90(im2aug, idxs_aug[2])
        count2aug = np.rot90(count2aug, idxs_aug[2])

        imgs_augment.append(im2aug)
        count_augment.append(count2aug)

    imgs_augment = np.array(imgs_augment)
    count_augment = np.array(count_augment)

    X_test = np.concatenate((img_raw[idxs_test, :, :, :], imgs_augment), axis=0)
    y_test = np.concatenate((count[idxs_test, :, :, :], count_augment), axis=0)

    X_data = np.concatenate((np.array(img_raw), X_test))
    y_data = np.concatenate((np.array(count), y_test))

    filename_data = 'dataset_64_1.h5'
    with h5py.File(filename_data, 'w') as f:
        f['raw'] = X_data
        f['count'] = y_data
        f['n_samples'] = img_raw.shape[0]
        f['flip_rot'] = flip_rot

def getCounts(labels,ef):
    pest=(np.sum(labels,axis=(1,2))/ef).astype(np.int)
    return pest

def summary_data(true_counts,pred_counts,X_test,y_test,y_pred,idx,obj,save_path):

    n_samples=true_counts.shape[0]

    #Predicted vs True counts for each image as histogram
    plt.figure(figsize=(25, 10))
    plt.bar(np.arange(n_samples), true_counts, width=0.4, align='center', label='True counts')
    plt.title('Counts for object '+str(obj+1))
    plt.bar(np.arange(n_samples), pred_counts, width=0.4, align='edge', label='Predicted counts')
    plt.legend()
    plt.xticks(np.arange(1,n_samples+1,10))
    plt.savefig(save_path + '/count_'+str(obj+1)+'.png')
    plt.close()

    #Regression target
    plt.figure()
    plt.Figure(figsize=(18, 9), dpi=160)
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[idx, :, :,0], cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(y_test[idx, :, :, obj],cmap='gray')
    plt.title('Target '+str(obj+1))

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[idx, :, :, 0], cmap='gray')
    plt.title('Predictions '+str(obj+1))

    plt.savefig(save_path+'/comparison_'+str(idx)+'_obj'+str(obj+1)+'.png')
    plt.close()

def getVideo(save_path):
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    width = 984
    height = 656
    video = cv2.VideoWriter(save_path + '/comparison.avi', fourcc, 10, (width, height))

    i = 0
    while True:
        file = save_path + 'comparisons/comp_{}.png'.format(i)
        if os.path.isfile(file):
            img = cv2.imread(file, 1)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            video.write(img)
            i = i + 1
        else:
            break

    video.release()

def getTrainingCurves(save_path):
    with open(save_path + 'train_curves.pickle', 'rb') as f:
        train_curves = pickle.load(f, encoding='bytes')

    plt.figure(figsize=(10,8))
    train_val = train_curves['train_val']
    test_val = train_curves['test_val']
    plt.plot(train_val, color='blue', label='Train loss')
    plt.plot(test_val, color='red', label='Test loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Mean loss over epoch(Training)")
    plt.ylim([0,10])
    plt.legend(loc=1)
    plt.savefig(save_path + '/train_curve.png')
    plt.close()

def getSummary(y_pred,save_path, idx,obj,filename_data):


    #Create a movie of the progress of training
    if os.path.exists(save_path + 'comparisons'):
        getVideo(save_path)

    #Plot training curves
    if os.path.isfile(save_path + 'train_curves.pickle'):
        getTrainingCurves(save_path)

    #Save hyper parameters as csv file
    with open(save_path + "hyper_param.pickle", 'rb') as f:
        hyper_param = pickle.load(f, encoding="bytes")

    patch_size = hyper_param['patch_size']
    stride = hyper_param['stride']
    idxs_train = hyper_param['idxs_train']
    idxs_test = hyper_param['idxs_test'] #Last 100 samples. Using that instead of idxs_test
    lr = hyper_param['lr']
    epochs = hyper_param['epochs']
    flip_rot = hyper_param['flip_rot']
    sz_tr = idxs_train.shape[0]

    with open(save_path + 'hyper_param.csv', mode='w') as hyp:
        hyp_writer = csv.writer(hyp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        hyp_writer.writerow(['Window size', patch_size])
        hyp_writer.writerow(['Stride', stride])
        hyp_writer.writerow(['Learning rate', lr])
        hyp_writer.writerow(['Epochs', epochs])
        hyp_writer.writerow(['Size of training data', str(sz_tr)])
        hyp_writer.writerow(['Training data', idxs_train])
        hyp_writer.writerow(['Test data', idxs_test])

    file_data = h5py.File(filename_data,'r')
    img_raw = file_data['raw'][()]
    count = file_data['count'][()]
    flip_rot = file_data['flip_rot'][()]

    #A bug fix.
    #For dead neurons 114 samples + data_aug to 250 samples
    #For live neurons first 14 samples + data_aug test data. Rest 100 images are training data.
    #So bug fix: Add test data to img_raw so that last 100 images are test data.
    if obj ==0:
        X_test = img_raw[idxs_test, :, :, :]
        y_test = count[idxs_test, :, :, :]
    elif obj==1:
        X_test = img_raw[-100:, :, :, :]
        y_test = count[-100:, :, :, :]

    ef = (patch_size / stride) ** 2

    assert (y_test.shape[0] == y_pred.shape[0])
    true_counts=getCounts(y_test,ef)[:,obj]
    pred_counts = getCounts(y_pred, ef)[:,0]

    #Plot comparison plots
    summary_data(true_counts,pred_counts,X_test,y_test,y_pred,idx,obj,save_path)

    #For training data set compute loss
    y_train = count[idxs_train,:,:,:]
    y_train_model = np.load(save_path+'train/train_predictions.npy')
    true_counts_train = getCounts(y_train, ef)[:, obj]
    pred_counts_train = getCounts(y_train_model, ef)[:,0]

    results_run = {}
    results_run['diff_counts'] = np.mean(np.abs(true_counts - pred_counts))
    results_run['n_train'] = sz_tr
    results_run['diff_counts_train'] = np.mean(np.abs(true_counts_train-pred_counts_train))

    return results_run

def getResults(save_folder,filename_data,obj,val_idx = 5):
    #Wrapper function for getting results for a particular run
    #save_folder: The path where the simulation is saved
    #obj: 0 or 1 for the dead or live neurons

    save_path = save_folder + 'target_{}/'.format(obj+1)
    if not os.path.exists(save_path):
        print('Path not found')
        print('Exiting ...')
        sys.exit(1)

    file_predictions = h5py.File(save_path + 'predictions.h5', 'r')
    y_pred = file_predictions['predictions'][()]
    if len(y_pred.shape)==3:
        y_pred = y_pred[:,:,:,None]

    #Wrapper to fix ambiguities
    if filename_data == 'dataset_64_1.h5' and not os.path.isfile(filename_data):
        # A bug fix.
        # For dead neurons 114 samples + data_aug to 250 samples
        # For live neurons first 14 samples + data_aug test data. Rest 100 images are training data.
        # So bug fix: Add test data to img_raw so that last 100 images are test data.
        fix_ambiguity()

    results_run = getSummary(y_pred,save_path, val_idx, obj,filename_data)

    return results_run


if __name__=='__main__':

    parser = argparse.ArgumentParser('Script to analyze the results of all simulations')

    parser.add_argument('--root_folder', default='//scratch-second/karthikp/', help='Root directory to save folder')
    parser.add_argument('--folderName', default='Jun_17/', help='Where to save folder')
    parser.add_argument('--filename_data', help='The augmented image data file', default='dataset_32_1.h5')
    parser.add_argument('--obj', default=0, type=int, help='0 for dead neurons and 1 for live neurons')

    args = parser.parse_args()

    root_folder = args.root_folder
    folderName = args.folderName
    filename_data = args.filename_data
    obj = args.obj

    save_folder = root_folder + folderName
    results_run = getResults(save_folder,filename_data,obj)

'''
import tensorflow as tf
from tensorflow.python.keras import models

save_folder = './Apr_1/sim_1/run_100'
filename_count = 'count_maps_64_1'
save_path = save_folder + '/target_2'
model = models.load_model(save_path + '/checkpoint/cc_model.h5')

X_train = img_raw[idxs_train,:,:,:]
y_model = model.predict(X_train)
#ytr_model = y_train_model[0,:,:,:][None,:,:,:]
y_train_model = np.load(save_path+'/train/train_predictions.npy')
epoch_loss = np.mean(np.abs(y_train_model - y_train))
y_test = y_test[:,:,:,1][:,:,:,None]
np.mean(np.abs(y_test - y_pred))
test_val[-1]
y_model.shape
ytr_model.shape

y_model.mean()
ytr_model.mean()


y_test.shape
y_pred.shape

true_counts = np.sum(y_test,axis=(1,2))/ef
pred_counts = np.sum(y_pred,axis=(1,2))/ef
'''
