import numpy as np
import h5py
import os
import argparse

class genData():
    def __init__(self,patch_size,stride,base_y,base_x,framesize_h=256,framesize_w=256):
        self.patch_size=patch_size
        self.stride=stride
        self.base_y=base_y
        self.base_x=base_x
        self.framesize_h=framesize_h
        self.framesize_w=framesize_w

    def getMarkersCells(self,labs):
        #labs: The annotated data set
        #First crop the dataset for the region considered
        #Second pad the markers with patch_size
        base_y=self.base_y
        base_x=self.base_x
        patch_size=self.patch_size
        framesize_h=self.framesize_h
        framesize_w=self.framesize_w
        markers = labs[base_y:base_y + framesize_h, base_x:base_x + framesize_w,:]
        npad=((patch_size,patch_size),(patch_size,patch_size),(0,0))
        markers = np.pad(markers, npad, "constant", constant_values=-1)
        return markers


    def getCellCountCells(self,markers, x,y,h,w):
        #Returns the no. of cells within a patch or a window
        #markers: Annotated dataset padded appropriately
        #(x,y): co-ordinates of the centre of the kernel
        #(h,w): height and width of the kernel

        noutputs=markers.shape[2]
        if(noutputs==3):
            noutputs=2
        counts = [0] * noutputs
        for i in range(noutputs):
            counts[i] = (markers[y:y+h,x:x+w,i] == 1).sum()
        return counts

    def getCountMap(self,markers, img_pad):

        base_y = self.base_y
        base_x = self.base_x
        patch_size = self.patch_size
        framesize_h = self.framesize_h
        framesize_w = self.framesize_w
        stride=self.stride

        height = int((img_pad.shape[0]) / stride)
        width = int((img_pad.shape[1]) / stride)
        noutputs = markers.shape[2]
        if(noutputs==3):
            noutputs=2
        count_map = np.zeros((height, width, noutputs))

        for y in range(0, height):
            for x in range(0, width):
                count = self.getCellCountCells(markers, x * stride, y * stride, patch_size, patch_size)
                for i in range(0, noutputs):
                    count_map[y][x][i] = count[i]

        count_total = self.getCellCountCells(markers, 0, 0, framesize_h + patch_size, framesize_w + patch_size)
        return count_map, count_total

    def getPadImage(self,img):
        base_y = self.base_y
        base_x = self.base_x
        patch_size = self.patch_size
        framesize_h = self.framesize_h
        framesize_w = self.framesize_w
        stride = self.stride

        #img is a grayscale image i.e. single channel
        assert len(img.shape)==2
        img_pad = img[base_y:base_y + framesize_h, base_x:base_x + framesize_w]
        img_pad = np.pad(img_pad, int((patch_size) / 2), "constant")
        return img_pad

    def getTrainingData(self,img_raw,annotated):

        base_y = self.base_y
        base_x = self.base_x
        patch_size = self.patch_size
        framesize_h = self.framesize_h
        framesize_w = self.framesize_w
        stride = self.stride

        n_samples=img_raw.shape[0]
        count_map_data=[]
        total_count=[]
        print("No. of samples: ",n_samples)
        for i in range(n_samples):
            print("Processing sample %d"%(i+1))
            markers = self.getMarkersCells(labs=annotated[i, :, :, :])
            img_pad = self.getPadImage(img=img_raw[i, :, :, 0])
            result=self.getCountMap(markers=markers, img_pad=img_pad)
            count_map_data.append(result[0])
            total_count.append(result[1])
        return np.array(count_map_data),np.array(total_count)


def genRandom(img, countMap):
    img_modify = img.copy()
    count_modify = countMap.copy()
    rot = np.random.randint(1, 5, size=1)[0]
    #print(rot)
    flip = np.random.randint(0, 2, size=1)[0]
    #print(flip)
    if (flip):
        img_modify = np.flipud(img_modify)
        count_modify = np.flipud(count_modify)

    img_modify = np.rot90(img_modify, rot)
    count_modify = np.rot90(count_modify, rot)
    if (flip != 0 or rot < 4):
        #print("Generated!")
        return [img_modify, count_modify, flip,rot]
    else:
        #print("Aborted!")
        return [None,None,None,None]


def data_augment(filename_count,filename_data,total_samples=200):
    #This function augments the data to make it upto total_samples
    #filename_count = 'count_maps_64_1'
    #filename_data = 'dataset_64_1.h5'

    #Set the seed for consistency

    np.random.seed(0)
    file_annotate = h5py.File('Neuron_annotated_dataset.h5','r')
    img_raw=file_annotate['raw']['data'][:10,...]

    if not os.path.isfile(filename_count):
        raise Exception("Error! File not found")

    file_count = h5py.File(filename_count, 'r')
    count_map_AMR = file_count['count_map_AMR']
    count_map_SG = file_count['count_map_SG']
    count_map_SI = file_count['count_map_SI']

    count = np.median((count_map_AMR,count_map_SG,count_map_SI),axis=0)
    n_samples = img_raw.shape[0]

    diff_len = total_samples - n_samples

    img_augment = []
    count_augment = []
    flip_augment =[]
    rot_augment =[]
    idxs_augment =[]
    sel = np.zeros(n_samples)
    while True:
        idx=np.random.choice(range(n_samples))
        if(sel[idx]==1):
            continue
        img_modify = img_raw[idx,:,:,:]
        count_modify = count[idx,:,:,:]
        img_modify,count_modify,flip,rot = genRandom(img=img_modify,countMap=count_modify)

        #Check for previous
        prev_idx = np.where(idxs_augment==idx)[0]
        if (len(prev_idx) > 0):
            prev_idx = prev_idx[0]
            if(flip==flip_augment[prev_idx] and rot==rot_augment[prev_idx]):
                continue
        if(img_modify is not None):

            sel[idx] = 1
            img_augment.append(img_modify)
            count_augment.append(count_modify)
            flip_augment.append(flip)
            rot_augment.append(rot)
            idxs_augment.append(idx)
        if(len(img_augment)==diff_len):
            break
        if(np.all(sel==1)):
            sel = np.zeros(n_samples)

    img_augment = np.stack(img_augment)
    count_augment = np.stack(count_augment)
    flip_augment = np.array(flip_augment)
    rot_augment = np.array(rot_augment)
    idxs_augment = np.array(idxs_augment)
    flip_rot = np.stack((idxs_augment,flip_augment,rot_augment),axis=-1)
    img_raw=np.concatenate((img_raw,img_augment),axis=0)
    count = np.concatenate((count,count_augment),axis=0)

    with h5py.File(filename_data,'w') as f:
        f['raw'] = img_raw
        f['count'] = count
        f['n_samples'] = n_samples
        f['flip_rot'] = flip_rot




if __name__ == '__main__':

    parser = argparse.ArgumentParser('Get Training data')
    parser.add_argument('--filename_annotate', default='Neuron_annotated_dataset.h5', help='Annotated dataset')
    parser.add_argument('--patch_size', default=32, type=int, help='Patch Size')
    parser.add_argument('--stride', default=1, type=int, help='Stride for patches')
    parser.add_argument('--filename_count', default='count_maps_32_1.h5', help='Name of count map file')
    parser.add_argument('--filename_data', default='dataset_32_1.h5', help='Name of augmented file used for training')
    args = parser.parse_args()

    filename_annotate = args.filename_annotate
    patch_size = args.patch_size
    stride = args.stride
    filename_count = args.filename_count
    filename_data = args.filename_data

    file_annotate = h5py.File(filename_annotate)
    imgs_raw = file_annotate['raw/data']
    gt_AMR = file_annotate['gt_AMR/data']
    gt_SG = file_annotate['gt_SG/data']
    gt_SI = file_annotate['gt_SI/data']

    data_cp = genData(patch_size=patch_size, stride=stride, base_x=0, base_y=0)

    count_map_AMR, _ = data_cp.getTrainingData(img_raw=imgs_raw, annotated=gt_AMR)
    count_map_SG, _ = data_cp.getTrainingData(img_raw=imgs_raw, annotated=gt_SG)
    count_map_SI, _ = data_cp.getTrainingData(img_raw=imgs_raw, annotated=gt_SI)

    with h5py.File(filename_count, 'w') as file_count:
        file_count['count_map_AMR'] = count_map_AMR
        file_count['count_map_SG'] = count_map_SG
        file_count['count_map_SI'] = count_map_SI

    data_augment(filename_count=filename_count,filename_data=filename_data,total_samples=250)