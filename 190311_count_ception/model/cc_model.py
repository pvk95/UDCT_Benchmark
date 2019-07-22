import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras import backend as K
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

class cc_model():
    def __init__(self, input_shape, patch_size, stride,save_folder,obj,lr=0.005, batch_sz=5, epochs=100, scope='cc_model'):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.stride = stride
        self.save_folder = save_folder
        self.ef = (self.patch_size / self.stride) ** 2
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.main_input = Input(shape=input_shape, name='main_input')
        self.main_output = []
        self.lr = lr
        self.obj = obj


        with tf.variable_scope(scope):
            self.model = self.create_model(save_folder)

    # custom layers (building blocks)
    def ConvFactory1(self, data, num_filters, filter_size, stride=1, pad=0, nonlinearity=LeakyReLU(alpha=0.3)):
        # data is the input tensor, leaky rely as a nonlinearity
        # the padding is done in the first layer automatically!
        # no need to preprocess the data
        data = ZeroPadding2D(padding=(pad, pad), data_format="channels_last", input_shape=self.input_shape)(data)
        data = Conv2D(filters=num_filters, kernel_size=(filter_size, filter_size), kernel_initializer='glorot_uniform')(
            data)
        data = LeakyReLU(alpha=0.3)(data)
        data = BatchNormalization()(data)
        return data

    def SimpleFactory1(self, data, ch_1x1, ch_3x3):
        # used for double layers
        conv1x1 = self.ConvFactory1(data, filter_size=1, pad=0, num_filters=ch_1x1)
        conv3x3 = self.ConvFactory1(data, filter_size=3, pad=1, num_filters=ch_3x3)
        concat = Concatenate()([conv1x1, conv3x3])
        return concat

    def create_model(self,save_folder):
        with open(save_folder + '/log.txt', 'a+') as f:
            f.write("\n===================>")
            net = self.ConvFactory1(self.main_input, num_filters=64, pad=self.patch_size, filter_size=3)
            # net = MaxPool2D()(net)
            f.write("\n" + str(net.shape))
            net = self.SimpleFactory1(net, ch_1x1=16, ch_3x3=16)
            f.write("\n" + str(net.shape))
            net = self.SimpleFactory1(net, ch_1x1=16, ch_3x3=32)
            f.write("\n" + str(net.shape))
            net = self.ConvFactory1(net, num_filters=16, filter_size=14)
            f.write("\n" + str(net.shape))
            net = self.SimpleFactory1(net, ch_1x1=112, ch_3x3=48)
            f.write("\n" + str(net.shape))
            net = self.SimpleFactory1(net, ch_1x1=40, ch_3x3=40)
            f.write("\n" + str(net.shape))
            net = self.SimpleFactory1(net, ch_1x1=32, ch_3x3=96)
            f.write("\n" + str(net.shape))

            net = self.ConvFactory1(net, num_filters=16, filter_size=18)
            f.write("\n" + str(net.shape))

            if self.obj ==0: #For dead neurons flt_sz = 1
                flt_sz = 1
            elif self.obj ==1: #For live neurons flt_sz = 17
                flt_sz = 17
            net = self.ConvFactory1(net, num_filters=64, filter_size=flt_sz)  # 17 changed to 1
            f.write("\n" + str(net.shape))
            net = self.ConvFactory1(net, num_filters=64, filter_size=flt_sz)  # 17 chnaged to 1
            f.write("\n" + str(net.shape))
            self.main_output = self.ConvFactory1(net, filter_size=1, num_filters=1)
            f.write("\n" + str(self.main_output.shape) + '\n')

            model = Model(inputs=[self.main_input], outputs=self.main_output)
            opt = tf.keras.optimizers.Adam(lr=self.lr)
            model.compile(loss = tf.keras.losses.mean_absolute_error, optimizer=opt)

        return model

    def save_model(self):
        model = self.model
        save_folder = self.save_folder
        tf.keras.models.save_model(model, save_folder + 'checkpoint/cc_model.h5')
        with open(save_folder + 'log.txt', 'a+') as f:
            f.write("\nModel saved in: {}".format(save_folder))

    def save_predictions(self,y_pred):
        save_folder = self.save_folder
        with h5py.File(save_folder + 'predictions.h5', 'w') as myFile:
            myFile['predictions'] = y_pred

        with open(save_folder + 'log.txt', 'a+') as f:
            f.write("\nPredictions saved in: {}".format(save_folder))

    def train_model(self, X_train, y_train, X_valid,y_valid,val_idx = 0):

        save_folder = self.save_folder
        tf.set_random_seed(0)
        np.random.seed(0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        img_valid = X_valid[val_idx, ...][None, ...]
        count_valid = y_valid[val_idx, ...]
        count_comp = 0

        n_samples = X_train.shape[0]

        if n_samples<self.batch_sz:
            batch_sz = n_samples
        else:
            batch_sz = self.batch_sz

        train_val = []
        test_val = []
        norm_val = [0]
        train_curves = {}

        epochs_lr = [self.lr] * self.epochs

        decay_rate = 4*np.log(2)/(self.epochs/5)

        for i in range(1, self.epochs):
            if (i < 3 * int(self.epochs / 5)):
                epochs_lr[i] = self.lr
            else:
                epochs_lr[i] = epochs_lr[i - 1] * np.exp(-decay_rate)

        # Create a checkpoint folder if not existent
        if not os.path.exists(save_folder + 'checkpoint'):
            os.makedirs(save_folder + 'checkpoint')


        for i in range(self.epochs):

            #Exponential decay the learning rate ( rate of decay set to 1e-2 set manually )
            current_lr = epochs_lr[i]
            K.set_value(self.model.optimizer.lr, current_lr)

            with open(save_folder + 'log.txt', 'a+') as f:
                f.write("\nEpoch {}/{}".format((i + 1), self.epochs))
            history = self.model.fit(x=X_train, y=y_train, batch_size=batch_sz, \
                                epochs=1, validation_split=0, verbose=False)

            #Visualize training after every epoch
            '''
            img_predict = model.predict(img_valid)
            self.visualize_train(img=img_valid, lab=count_valid, pcount=img_predict, \
                                 save_folder=save_folder, idx=count_comp)
            count_comp = count_comp + 1
            '''

            train_val.append(history.history['loss'][-1])
            with open(save_folder + 'log.txt', 'a+') as f:
                f.write("\nMean loss over epoch: "+ str(train_val[-1]))

            y_pred = self.model.predict(X_valid,batch_size=16)
            test_val_epoch = np.mean(np.abs(y_pred - y_valid))
            test_val.append(test_val_epoch)

            with open(save_folder + 'log.txt', 'a+') as f:
                f.write("\nTest loss: "+ str(test_val[-1]))

            train_curves['train_val'] = np.array(train_val)
            train_curves['test_val'] = np.array(test_val)
            train_curves['grad_norm'] = np.array(norm_val)

            with open(save_folder + "train_curves.pickle", 'wb') as f:
                pickle.dump(train_curves, f)

            # Save model after every 10th epoch and make predictions
            if ((i + 1) % 10 == 0 or i==self.epochs -1):
                self.save_model()

            #Save predictions on test data after every 100 epochs
            if ((i+1) % 100 ==0 or i==self.epochs-1):
                self.save_predictions(y_pred)

            #Make predictions on train data every 250 epochs
            if ((i+1) % 250 ==0 or i==self.epochs-1):
                if not os.path.exists(save_folder + 'train'):
                    os.makedirs(save_folder + 'train')
                y_pred_train = self.model.predict(X_train,batch_size = 16)
                np.save(save_folder + 'train/train_predictions', y_pred_train)

        return self.model

    def visualize_train(self, img=None, lab=None, pcount=None, save_folder=None, idx=None):

        # img is the ground truth image
        # lab is the target count map of img
        # pcount is the predicted count map

        img = np.squeeze(img)
        pcount = np.reshape(pcount, [pcount.shape[1], pcount.shape[2], -1])
        fig = plt.Figure(figsize=(6, 4), dpi=164)
        gcf = plt.gcf()
        gcf.set_size_inches(6, 4)

        ax2 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
        ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=3)
        ax4 = plt.subplot2grid((2, 4), (1, 2), colspan=3)
        ax5 = plt.subplot2grid((2, 4), (1, 0), rowspan=1)
        ax6 = plt.subplot2grid((2, 4), (1, 1), rowspan=1)

        ax2.set_title("Input Image")
        ax2.imshow(img, interpolation='none', cmap='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3.set_title("Regression target, {}x{}".format(self.patch_size, self.patch_size))
        temp = []
        for i in range(lab.shape[2]):
            temp.append(lab[:, :, i])

        ax3.imshow(np.hstack(temp), interpolation='none')
        ax3.set_xticks([])
        ax3.set_yticks([])
        del temp

        ax4.set_title("Predicted counts")
        temp = []
        for i in range(pcount.shape[2]):
            temp.append(pcount[:, :, i])
        ax4.imshow(np.hstack(temp), interpolation='none')
        ax4.set_xticks([])
        ax4.set_yticks([])
        del temp

        pred_est = (np.sum(pcount, axis=(0, 1)) / self.ef).astype(np.int)
        lab_est = (np.sum(lab, axis=(0, 1)) / self.ef).astype(np.int)
        noutputs = lab.shape[2]

        ax5.set_title("Real:" + str(lab_est))
        ax5.set_ylim((0, np.max(lab_est) * 2))
        ax5.set_xticks(np.arange(0, noutputs, 1.0))
        ax5.bar(np.arange(noutputs), lab_est, align='center')
        ax6.set_title("Pred:" + str(pred_est))
        ax6.set_ylim((0, np.max(lab_est) * 2))
        ax6.set_xticks(np.arange(0, noutputs, 1.0))
        ax6.set_yticks([])
        ax6.bar(np.arange(noutputs), pred_est, align='center')

        save_path = save_folder + '/comparisons/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + 'comp_' + str(idx) + '.png', dpi=164)


    def getPredictions(self,X_test):

        save_folder = self.save_folder
        if not os.path.isfile(save_folder+'checkpoint/cc_model.h5'):
            with open(save_folder + 'log.txt', 'a+') as f:
                f.write("Model not found! \nExiting ...\n")
            sys.exit(1)

        model = tf.keras.models.load_model(save_folder+'/checkpoint/cc_model.h5')
        y_pred = model.predict(X_test,batch_size=16)
        self.save_predictions(y_pred)
        return y_pred

if __name__=='main':

    init_lr = 0.005
    n_epochs = 500
    epochs = np.arange(n_epochs)
    epochs_lr = [0.01] * n_epochs

    k = 4*np.log(2)/(n_epochs/5)

    for i in range(1, n_epochs):
        if (i < 3 * int(n_epochs / 5)):
            epochs_lr[i] = 0.01
        else:
            epochs_lr[i] = epochs_lr[i - 1] * np.exp(-k)

    plt.plot(epochs,epochs_lr)

'''
#Following version: When the network is updated for every batch.

    def train_model(self, X_train, y_train, img_valid,count_valid,X_valid,y_valid,save_folder="cc_model_results"):

        tf.set_random_seed(0)
        np.random.seed(0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        img_valid = img_valid[None, :, :, :]

        n_samples = X_train.shape[0]

        train_val = []
        test_val=[]
        norm_val=[0]
        train_curves = {}

        count_comp=0

        epochs_lr = [self.lr] * self.epochs

        decay_rate = 4 * np.log(2) / (self.epochs / 5)

        for i in range(1, self.epochs):
            if (i < 3 * int(self.epochs / 5)):
                epochs_lr[i] = self.lr
            else:
                epochs_lr[i] = epochs_lr[i - 1] * np.exp(-decay_rate)

        # Create a checkpoint folder if not existent
        if not os.path.exists(save_folder + '/checkpoint'):
            os.makedirs(save_folder + '/checkpoint')
        checkpoint_path = save_folder + "/checkpoint/cc_model.ckpt"

        epoch_val = []

        model = self.model
        if n_samples<self.batch_sz:
            batch_sz = n_samples
        else:
            batch_sz = self.batch_sz

        for i in range(self.epochs):

            current_lr = epochs_lr[i]
            K.set_value(model.optimizer.lr, current_lr)

            idxs = np.random.choice(np.arange(n_samples),size = batch_sz,replace = False)
            xtr = X_train[idxs,:,:,:]
            ytr = y_train[idxs, :, :, :]

            mae_loss = model.train_on_batch(xtr,ytr)
            epoch_val.append(mae_loss)

            if((i+1)%self.epochs%30==0 or i==self.epochs-1):


                #img_predict = model.predict(img_valid)
                #self.visualize_train(img=img_valid, lab=count_valid, pcount=img_predict, save_folder=save_folder,idx=count_comp)

                #count_comp = count_comp + 1

                train_val.append(np.mean(epoch_val))
                epoch_val = []

                y_pred = model.predict(X_valid,batch_size = 16)
                test_val_epoch = np.mean(np.abs(y_valid-y_pred))
                test_val.append(test_val_epoch)

                with open(save_folder + '/log.txt', 'a+') as f:
                    f.write("\nEpoch {}/{}".format((i + 1), self.epochs))
                    f.write("\nSmoothed mean loss: {}".format(train_val[-1]))
                    f.write("\nTest loss: {}".format(test_val[-1]))

                train_curves['train_val'] = np.array(train_val)
                train_curves['test_val'] = np.array(test_val)
                train_curves['grad_norm'] = np.array(norm_val)

                with open(save_folder + "/train_curves.pickle", 'wb') as f:
                    pickle.dump(train_curves,f)

                #Save model after every 10th epoch and make predictions
                if((i+1)%300==0 or i==self.epochs-1):
                    tf.keras.models.save_model(model,save_folder+'/checkpoint/cc_model.h5')
                    with open(save_folder + '/log.txt', 'a+') as f:
                        f.write("\nModel saved in: {}".format(save_folder))

                #Save test predictions every 100 epochs
                if((i+1)%3000==0 or i==self.epochs-1):

                    file_predict = save_folder + '/predictions.h5'
                    f = h5py.File(file_predict, 'w')
                    f['predictions'] = y_pred
                    f.close()
                    with open(save_folder + '/log.txt', 'a+') as f:
                        f.write("\nPredictions saved in: {}".format(save_folder))

                #Save train predictions every 250 epochs
                if ((i+1)%7500==0 or i==self.epochs-1):
                    if not os.path.exists(save_folder + '/train'):
                        os.makedirs(save_folder + '/train')
                    feed_train_valid = {self.main_input: X_train}
                    y_pred_train = model.predict(X_train,batch_size = 16)
                    np.save(save_folder + '/train/train_predictions', y_pred_train)
'''