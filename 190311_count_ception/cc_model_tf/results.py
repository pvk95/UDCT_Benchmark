import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
import cv2
import os
import csv
import pickle

def getCounts(labels,ef):
    pest=(np.sum(labels,axis=(1,2))/ef).astype(np.int)
    return pest

def summary_data(true_counts,pred_counts,X_test,y_test,y_pred,idx,obj,save_path):

    n_samples=true_counts.shape[0]

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
    if y_pred.shape[3]==2:
        plt.imshow(y_pred[idx, :, :, obj], cmap='gray')
    else:
        plt.imshow(y_pred[idx, :, :, 0], cmap='gray')
    plt.title('Predictions '+str(obj+1))
    plt.savefig(save_path+'/comparison_'+str(idx)+'_obj'+str(obj+1)+'.png')
    plt.close()


def getSummary(y_pred,save_path, idx,obj,filename_count):

    if os.path.exists(save_path + '/comparisons'):

        #Create a movie
        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        width = 984
        height = 656
        video = cv2.VideoWriter(save_path + '/comparison.avi', fourcc, 10, (width, height))

        i = 0
        while True:
            file = save_path + '/comparisons/comp_' + str(i) + '.png'
            if os.path.isfile(file):
                img = cv2.imread(file, 1)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                video.write(img)
                i = i + 1
            else:
                break

        video.release()

    #Plot training curves
    if os.path.isfile(save_path + '/train_curves.pickle'):
        with open(save_path + '/train_curves.pickle', 'rb') as f:
            train_curves = pickle.load(f, encoding='bytes')

    plt.figure()
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

    #Save hyper parameters as csv file
    with open(save_path + "/hyper_param.pickle", 'rb') as f:
        hyper_param = pickle.load(f,encoding="bytes")

    patch_size = hyper_param['patch_size']
    stride = hyper_param['stride']
    idxs_train = hyper_param['idxs_train']
    idxs_test = hyper_param['idxs_test']
    lr = hyper_param['lr']
    epochs = hyper_param['epochs']
    flip_rot = hyper_param['flip_rot']
    sz_tr = idxs_train.shape[0]


    with open(save_path + '/hyper_param.csv', mode='w') as hyp:
        hyp_writer = csv.writer(hyp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        hyp_writer.writerow(['Window size', patch_size])
        hyp_writer.writerow(['Stride', stride])
        hyp_writer.writerow(['Learning rate', lr])
        hyp_writer.writerow(['Epochs', epochs])
        hyp_writer.writerow(['Size of training data', str(sz_tr)])
        hyp_writer.writerow(['Training data',idxs_train])
        hyp_writer.writerow(['Test data',idxs_test])

    '''
    filename_annotate = 'Neuron_annotated_dataset.h5'
    file_annotate = h5py.File(filename_annotate, 'r')
    img_raw = file_annotate['raw']['data'][()]

    file_count = h5py.File(filename_count, 'r')
    count_map_AMR = file_count['count_map_AMR'][()]
    count_map_SG = file_count['count_map_SG'][()]
    count_map_SI = file_count['count_map_SI'][()]

    count = np.median((count_map_AMR,count_map_SG,count_map_SI),axis=0)

    ef = (patch_size / stride) ** 2

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
    '''

    file_data = h5py.File('dataset_32_1.h5','r')
    img_raw = file_data['raw'][()]
    count = file_data['count'][()]
    flip_rot = file_data['flip_rot'][()]
    X_test = img_raw[idxs_test, :, :, :]
    y_test = count[idxs_test, :, :, :]
    ef = (patch_size / stride) ** 2


    assert (y_test.shape[0] == y_pred.shape[0])
    true_counts=getCounts(y_test,ef)[:,obj]
    pred_counts = getCounts(y_pred, ef)

    if pred_counts.shape[1]==1:
        pred_counts=pred_counts[:,0]
    else:
        pred_counts=pred_counts[:,obj]

    #Plot comparison plots
    summary_data(true_counts,pred_counts,X_test,y_test,y_pred,idx,obj,save_path)

    #For training data set compute loss
    y_train = count[idxs_train,:,:,:]
    y_train_model = np.load(save_path+'/train/train_predictions.npy')
    true_counts_train = getCounts(y_train, ef)[:, obj]
    pred_counts_train = getCounts(y_train_model, ef)

    if pred_counts_train.shape[1] == 1:
        pred_counts_train = pred_counts_train[:, 0]
    else:
        pred_counts_train = pred_counts_train[:, obj]

    results_run = {}
    results_run['diff_counts'] = np.mean(np.abs(true_counts - pred_counts))
    results_run['n_train'] = sz_tr
    results_run['diff_counts_train'] = np.mean(np.abs(true_counts_train-pred_counts_train))

    return results_run

def getResults(save_folder):

    if os.path.exists(save_folder + '/predictions.h5'):
        save_path=save_folder
        file_predictions = h5py.File(save_path + '/predictions.h5', 'r')
        y_pred = file_predictions['predictions']
        getSummary(y_pred,save_path,5,0,'count_maps_64_1')
        getSummary(y_pred, save_path,5, 1,'count_maps_64_1')
    else:
        if os.path.exists(save_folder+'target_1'):
            print("Analyzing results for target 1 ===>")
            save_path=save_folder + 'target_1'
            file_predictions_1 = h5py.File(save_path + '/predictions.h5', 'r')
            y_pred = file_predictions_1['predictions'][()]
            if len(y_pred.shape)==3:
                y_pred = y_pred[:,:,:,None]
            results_run = getSummary(y_pred,save_path, 5, 0,'count_maps_32_1')
            print("Analysis for target 1 complete")
        elif os.path.exists(save_folder+'target_2'):
            print("Analyzing results for target 2 ===>")
            save_path = save_folder + 'target_2'
            file_predictions_2 = h5py.File(save_path + '/predictions.h5', 'r')
            y_pred = file_predictions_2['predictions'][()]
            results_run = getSummary(y_pred,save_path, 5, 1,'count_maps_64_1')
            print("Analysis for target 2 complete")
        else:
            print("Analyzing results for unknown object")
            print("Exiting ...")
            sys.exit(1)

    return results_run


if __name__=='__main__':
    save_folder = './Apr_15/'
    save_folder = sys.argv[1]
    dict = getResults(save_folder)

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
