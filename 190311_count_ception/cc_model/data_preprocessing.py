import numpy as np

def getMarkersCells(labs,base_y,base_x,patch_size=32,framesize_h=256,framesize_w=256):
    #labs: The annotated data set
    #First crop the dataset for the region considered
    #Second pad the markers with patch_size
    markers = labs[base_y:base_y + framesize_h, base_x:base_x + framesize_w,:]
    npad=((patch_size,patch_size),(patch_size,patch_size),(0,0))
    markers = np.pad(markers, npad, "constant", constant_values=-1)
    return markers


def getCellCountCells(markers, x,y,h,w):
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

def getCountMap(markers, img_pad,stride=1,patch_size=32,framesize_h=256,framesize_w=256):

    height = int((img_pad.shape[0]) / stride)
    width = int((img_pad.shape[1]) / stride)
    noutputs = markers.shape[2]
    if(noutputs==3):
        noutputs=2
    count_map = np.zeros((height, width, noutputs))

    for y in range(0, height):
        for x in range(0, width):
            count = getCellCountCells(markers, x * stride, y * stride, patch_size, patch_size)
            for i in range(0, noutputs):
                count_map[y][x][i] = count[i]

    count_total = getCellCountCells(markers, 0, 0, framesize_h + patch_size, framesize_w + patch_size)
    return count_map, count_total

def getPadImage(img,base_y=0,base_x=0,patch_size=32,framesize_h=256,framesize_w=256):
    #img is a grayscale image i.e. single channel
    assert len(img.shape)==2
    img_pad = img[base_y:base_y + framesize_h, base_x:base_x + framesize_w]
    img_pad = np.pad(img_pad, int((patch_size) / 2), "constant")
    return img_pad

def getTrainingData(img_raw,annotated):
    n_samples=img_raw.shape[0]
    count_map_data=[]
    total_count=[]
    print("No. of samples: ",n_samples)
    for i in range(n_samples):
        print("Processing sample %d"%(i+1))
        markers = getMarkersCells(labs=annotated[i, :, :, :], base_y=0, base_x=0)
        img_pad = getPadImage(img_raw[i, :, :, 0])
        result=getCountMap(markers=markers, img_pad=img_pad, stride=stride)
        count_map_data.append(result[0])
        total_count.append(result[1])
    return np.array(count_map_data),np.array(total_count)
