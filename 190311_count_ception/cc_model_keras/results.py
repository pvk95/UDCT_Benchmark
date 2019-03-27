import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
import cv2
import os
import csv
import pickle

save_folder=sys.argv[1]
#save_folder='22_Mar'

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
    plt.close()

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
    plt.close()


def getSummary(y_pred,save_path, idx,obj):

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

    '''
    plt.figure()
    norm_val = train_curves['grad_norm']
    plt.plot(norm_val, color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Grad norm")
    plt.savefig(save_path + '/grad_norm.png')
    '''

    #Save hyper parameters as csv file
    with open(save_path + "/hyper_param.pickle", 'rb') as f:
        hyper_param = pickle.load(f,encoding="bytes")

    patch_size = hyper_param['patch_size']
    stride = hyper_param['stride']
    idxs_train = hyper_param['idxs_train']
    idxs_test = hyper_param['idxs_test']
    lr = hyper_param['lr']
    epochs = hyper_param['epochs']
    combine = hyper_param['combine']

    print("Patch Size: ",patch_size)
    print("Stride: ", stride)
    print("idxs_train: ", idxs_train)
    print("idxs_test: ", idxs_test)
    print("Learning rate: ", lr)
    print("Epochs: ", epochs)

    filename_annotate = 'Neuron_annotated_dataset.h5'
    file_annotate = h5py.File(filename_annotate, 'r')

    file_count = h5py.File('count_maps_64_1', 'r')
    count_map_AMR = file_count['count_map_AMR'][()][idxs_test, :, :, :]
    count_map_SG = file_count['count_map_SG'][()][idxs_test, :, :, :]
    count_map_SI = file_count['count_map_SI'][()][idxs_test, :, :, :]

    count = np.median((count_map_AMR,count_map_SG,count_map_SI),axis=0)

    ef = (patch_size / stride) ** 2

    true_counts=getCounts(count,ef)[:,obj]
    pred_counts = getCounts(y_pred, ef)
    if pred_counts.shape[1]==1:
        pred_counts=pred_counts[:,0]
    else:
        pred_counts=pred_counts[:,obj]

    img_raw = file_annotate['raw']['data'][()][idxs_test, :, :, :]

    with open(save_path + '/hyper_param.csv', mode='w') as hyp:
        hyp_writer = csv.writer(hyp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        hyp_writer.writerow(['Window size', patch_size])
        hyp_writer.writerow(['Stride', stride])
        hyp_writer.writerow(['Learning rate', lr])
        hyp_writer.writerow(['Epochs', epochs])
        hyp_writer.writerow(['Combined model?', str(combine)])

    #Plot comparison plots
    summary_data(true_counts,pred_counts,img_raw,count,idx,obj,save_path)

if os.path.exists(save_folder + '/predictions.h5'):
    save_path=save_folder
    file_predictions = h5py.File(save_path + '/predictions.h5', 'r')
    y_pred = file_predictions['predictions']
    getSummary(y_pred,save_path,5,0)
    getSummary(y_pred, save_path,5, 1)
elif os.path.exists(save_folder+'/target_1'):
    save_path=save_folder + '/target_1'
    file_predictions_1 = h5py.File(save_path + '/predictions.h5', 'r')
    y_pred = file_predictions_1['predictions'][()]
    if len(y_pred.shape)==3:
        y_pred=y_pred[:,:,:,None]
    getSummary(y_pred,save_path, 5, 0)
elif os.path.exists(save_folder+'/target_2'):
    save_path = save_folder + '/target_2'
    file_predictions_2 = h5py.File(save_path + '/predictions.h5', 'r')
    y_pred = file_predictions_2['predictions'][()]
    if len(y_pred.shape)==3:
        y_pred=y_pred[:,:,:,None]
    getSummary(y_pred,save_path, 5, 1)
else:
    print("Analyzing results for unknown object")
    print("Exiting ...")
    sys.exit(1)
