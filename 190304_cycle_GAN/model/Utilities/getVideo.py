import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

if __name__=='__main__':
    save_folder = sys.argv[1]
    save_path = save_folder + 'Images/'

    im = plt.imread(save_path+'CGAN_Epoch_1.png')
    height,width = im.shape[:2]
    del im

    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    video = cv2.VideoWriter(save_folder + 'ImagesGen.avi', fourcc, 10, (width, height))

    i = 1
    while True:
        file = save_path + 'CGAN_Epoch_' + str(i) + '.png'
        if os.path.isfile(file):
            img = cv2.imread(file, 1)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            video.write(img)
            i = i + 1
        else:
            break

    video.release()
