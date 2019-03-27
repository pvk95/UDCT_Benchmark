import matplotlib.pyplot as plt
import h5py
import numpy as np
import pickle
import sys

save_folder=sys.argv[1]
pickle_out = open(save_folder+"hyper_param.pickle",'rb')
hyper_param = pickle.load(pickle_out)
pickle_out.close()

patch_size=hyper_param['patch_size']
stride=hyper_param['stride']
idxs_train=hyper_param['idxs_train']
idxs_test=hyper_param['idxs_test']
lr=hyper_param['lr']
epochs=hyper_param['epochs']
combine=hyper_param['combine']
#lr=0.005
#epochs=100
#combine=False

if(combine):
    file_predictions = h5py.File(save_folder + 'predictions.h5', 'r')
    y_pred = file_predictions['predictions']
else:
    file_predictions_1 = h5py.File(save_folder+'target_1/predictions.h5','r')
    file_predictions_2 = h5py.File(save_folder+'target_2/predictions.h5','r')
    y_pred_1=file_predictions_1['predictions']
    #y_pred_1=np.zeros((12,288,288))
    y_pred_2=file_predictions_2['predictions']
    y_pred=np.stack((y_pred_1,y_pred_2),axis=-1)

n_samples=y_pred.shape[0]
n_outputs=y_pred.shape[3]

filename_annotate='Neuron_annotated_dataset.h5'
file_annotate=h5py.File(filename_annotate, 'r')
img_raw=file_annotate['raw']['data'][idxs_test,:,:,:]

file_count = h5py.File('count_maps', 'r')
count_map_AMR = file_count['count_map_AMR'][idxs_test,:,:,:]

ef=(patch_size/stride)**2

def getCounts(labels):
    pest=(np.sum(labels,axis=(1,2))/ef).astype(np.int)
    return pest

true_counts=getCounts(count_map_AMR)
pred_counts=getCounts(y_pred)


def summary_data(idx):
    print("No. of samples: ", img_raw.shape[0])
    print("Raw image shape: ", img_raw[0, :, :, 0].shape)
    print("No. of channels: ",img_raw.shape[3])

    print("Count map size: ",count_map_AMR.shape[1:])
    print("Learning rate: ",lr)
    print("Epochs: ", epochs)

    if(combine):
        print("Model trained combinedly")
    else:
        print("Model trained separately")

    plt.figure()
    plt.bar(np.arange(n_samples), true_counts[:, 0], align='center', label='True counts')
    plt.title('Counts for object 1')
    plt.bar(np.arange(n_samples), pred_counts[:, 0], align='center', label='Predicted counts')
    plt.legend()
    plt.savefig(save_folder + 'count_1.png')

    plt.figure()
    plt.bar(np.arange(n_samples), true_counts[:, 1], align='center', label='True counts')
    plt.title('Counts for object 2')
    plt.bar(np.arange(n_samples), pred_counts[:, 1], align='center', label='Predicted counts')
    plt.legend()
    plt.savefig(save_folder+'count_2.png')

    #Regression target 1
    plt.figure()
    plt.Figure(figsize=(18, 9), dpi=160)
    plt.subplot(1, 3, 1)
    plt.imshow(img_raw[idx, :, :,0], cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(count_map_AMR[idx, :, :, 0],cmap='gray')
    plt.title('Target 1')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[idx, :, :, 0], cmap='gray')
    plt.title('Predictions 1')
    plt.savefig(save_folder+'comparison_'+str(idx)+'_obj1.png')

    #Regression target 2
    plt.figure()
    plt.Figure(figsize=(18, 9), dpi=160)
    plt.subplot(1, 3, 1)
    plt.imshow(img_raw[idx, :, :, 0], cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(count_map_AMR[idx, :, :, 1],cmap='gray')
    plt.title('Target 2')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[idx, :, :, 1],cmap='gray')
    plt.title('Predictions 2')
    plt.savefig(save_folder + 'comparison_' + str(idx) + '_obj2.png')

summary_data(5)

#####################################
#VGG Cell data set
'''
import glob
img_raw=[]

for cell in glob.glob('cells/*cell.png'):
    im=plt.imread(cell)
    img_raw.append(im)
img_raw = np.array(img_raw)

markers=[]
for mk in glob.glob('cells/*dots.png'):
    im=plt.imread(mk)
    markers.append(im)
markers=np.array(markers)
'''

def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)

def h5Hieararchy(file):
    with h5py.File(file, 'r') as f:
        print('Hierarchy of file: ')
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)