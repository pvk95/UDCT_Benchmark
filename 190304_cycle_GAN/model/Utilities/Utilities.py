import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
sys.path.append('./model/')
import results

def produce_tiled_images(im_A,im_B,fake_A,fake_B,cyc_A,cyc_B):

    list_of_images=[im_A,im_B,fake_A,fake_B,cyc_A,cyc_B]
    for i in range(6):
        if np.shape(list_of_images[i])[-1]==1:
            list_of_images[i]=np.tile(list_of_images[i],[1,1,1,3])  
        list_of_images[i]=np.pad(list_of_images[i][0,:,:,:], ((20,20),(20,20),(0,0)), mode='constant', constant_values=[0.5])
    im_A,im_B,fake_A,fake_B,cyc_A,cyc_B=list_of_images
    a=np.vstack( (im_A,im_B))
    b=np.vstack( (fake_B,fake_A))
    c=np.vstack( (cyc_A,cyc_B))
    return np.hstack((a,b,c))

def visualize_train(im,gt,fake,path = None):

    assert len(im.shape) ==2 #Grayscale image
    assert len(gt.shape) ==2 #Only one channel expected and squeezed
    assert len(fake.shape) ==3 #All 3 channels expected
    fake_gt, _ = results.get_location_dead(fake)

    assert len(fake_gt.shape) == 2 #Only one channel expected

    true_counts = np.sum(gt)
    pred_counts = np.sum(fake_gt)

    fig = plt.Figure(figsize=(12,8),dpi=200)
    gcf = plt.gcf()
    gcf.set_size_inches(12,8)

    ax1 = plt.subplot2grid((2,4),(0,0),colspan=2) # Image to be segmented (labelled with dead neurons)
    ax2 = plt.subplot2grid((2,4),(0,2),colspan=2) # Segmented image
    ax3 = plt.subplot2grid((2,4),(1,2),colspan=2) # Image with predicted positions of dead neurons
    ax4 = plt.subplot2grid((2,4),(1,0)) #Number of neurons
    ax5 = plt.subplot2grid((2,4),(1,1)) #Number of predicted neurons

    ax1.imshow(gt + im,cmap='gray')
    ax1.set_title('Labelled image')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(fake)
    ax2.set_title('Segmented image')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(fake_gt + im , cmap = 'gray')
    ax3.set_title('Predicted image')
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4.bar(np.arange(1),true_counts,align = 'center')
    ax4.set_title('True counts: ' + str(true_counts))
    ax4.set_ylim(0,true_counts*2)
    ax4.set_xticks(np.arange(1))

    ax5.bar(np.arange(1), pred_counts, align='center')
    ax5.set_title('Predicted counts: ' + str(pred_counts))
    ax5.set_ylim(0, true_counts * 2)
    ax5.set_xticks(np.arange(1))

    plt.savefig(path,dpi = 200)
    plt.close(fig)

if __name__ =='__main__':
    fileName_data = '../190311_count_ception/Neuron_annotated_dataset.h5'
    save_folder = 'run_1585/'
    file_pred = h5py.File(save_folder + 'CGANdata_gen_B.h5','r')
    y_pred = file_pred['B/data']

    if y_pred.dtype == 'uint16':
        y_pred = np.array(y_pred) / (2 ** 16 - 1)

    file_gt = h5py.File(fileName_data,'r')
    imgs_raw = file_gt['raw/data']
    gt_AMR = file_gt['gt_AMR/data']
    gt_SG = file_gt['gt_SG/data']
    gt_SI = file_gt['gt_SI/data']

    im = np.array(imgs_raw[0,:,:,0])
    gt = np.array(gt_SI[0,:,:,0])
    fake = y_pred[0,:,:,:]

    visualize_train(im,gt,fake)
