from __future__ import division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.regularizers import l1
from tensorflow.python.keras.callbacks import History
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle

class cc_model():
    def __init__(self, input_shape, patch_size, stride,lr=0.005, batch_sz=5, epochs=100, scope='cc_model'):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.stride=stride
        self.ef=(self.patch_size/self.stride)**2
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.main_input = Input(shape=input_shape, name='main_input')
        self.main_output = []
        self.lr = lr

        with tf.variable_scope(scope):
            self.model = self.create_model()

    def getPlaceholders(self):
        return [self.main_input,self.main_output]

    # custom loss funciton
    def custom_mae_loss(self,y_true, y_pred):
        # mae_loss might be too "greedy" and train the network on artifacts
        # prediction_count2 = np.sum(y_pred / ef)
        # mae_loss = K.sum(K.abs(prediction_count2 - (y_true/ef)))
        # Mean Absolute Error is computed between each count of the count map

        l1_loss = tf.abs(y_true - y_pred)
        #loss = tf.math.reduce_mean(l1_loss,axis=0)
        #loss = tf.reduce_sum(loss)/self.ef
        loss = tf.reduce_mean(l1_loss)
        return loss

    # custom layers (building blocks)
    def ConvFactory1(self, data, num_filters, filter_size, stride=1, pad=0, nonlinearity=LeakyReLU(alpha=0.3)):
        # data is the input tensor, leaky rely as a nonlinearity
        # the padding is done in the first layer automatically!
        # no need to preprocess the data
        data = ZeroPadding2D(padding=(pad, pad), data_format="channels_last", input_shape=self.input_shape)(data)
        data = Conv2D(filters=num_filters, kernel_size=(filter_size, filter_size), kernel_initializer='glorot_uniform')(data)
        data = LeakyReLU(alpha=0.3)(data)
        data = BatchNormalization()(data)
        #data = tf.keras.layers.Dropout(0.2)(data)
        return data

    def SimpleFactory1(self, data, ch_1x1, ch_3x3):
        # used for double layers
        conv1x1 = self.ConvFactory1(data, filter_size=1, pad=0, num_filters=ch_1x1)
        conv3x3 = self.ConvFactory1(data, filter_size=3, pad=1, num_filters=ch_3x3)
        concat = Concatenate()([conv1x1, conv3x3])
        return concat

    def create_model(self):
        net = self.ConvFactory1(self.main_input, num_filters=64, pad=self.patch_size, filter_size=3)
        #net = MaxPool2D()(net)
        print(net.shape)
        net = self.SimpleFactory1(net, ch_1x1=16, ch_3x3=16)
        print(net.shape)
        net = self.SimpleFactory1(net, ch_1x1=16, ch_3x3=32)
        print(net.shape)
        net = self.ConvFactory1(net, num_filters=16, filter_size=14)
        print(net.shape)
        net = self.SimpleFactory1(net, ch_1x1=112, ch_3x3=48)
        print(net.shape)
        net = self.SimpleFactory1(net, ch_1x1=40, ch_3x3=40)
        print(net.shape)
        net = self.SimpleFactory1(net, ch_1x1=32, ch_3x3=96)
        print(net.shape)

        net = self.ConvFactory1(net, num_filters=16, filter_size=18)
        print(net.shape)
        net = self.ConvFactory1(net, num_filters=64, filter_size=1)
        print(net.shape)
        net = self.ConvFactory1(net, num_filters=64, filter_size=1)
        print(net.shape)
        self.main_output = self.ConvFactory1(net, filter_size=1, num_filters=2)
        print(self.main_output.shape)

        model = Model(inputs=[self.main_input], outputs=self.main_output)
        opt = tf.keras.optimizers.Adam(lr=self.lr)
        mae_loss = tf.keras.losses.mean_absolute_error
        model.compile(loss=mae_loss, optimizer=opt)

        return model


    def train_model(self, X_train, y_train, img_valid,count_valid,X_valid,y_valid,save_folder="cc_model_results"):

        tf.set_random_seed(0)
        np.random.seed(0)

        img_valid = img_valid[None,:,:,:]

        train_val = []
        test_val = []
        norm_val = [0]

        train_curves = {}

        count_comp = 0

        # Create a checkpoint folder if not existent
        if not os.path.exists(save_folder + '/checkpoint'):
            os.makedirs(save_folder + '/checkpoint')
        checkpoint_path = save_folder + "/checkpoint/cc_model.ckpt"

        model = self.model
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, period=10)


        for i in range(self.epochs):
            print("\nEpoch {}/{}".format((i + 1), self.epochs))
            history=model.fit(x=X_train, y=y_train, batch_size=self.batch_sz, epochs=1,\
                      validation_split=0,callbacks=[cp_callback],verbose=True)

            img_predict = model.predict(img_valid)
            self.visualize_train(img=img_valid, lab=count_valid, pcount=img_predict, \
                                 save_folder=save_folder,idx=count_comp)

            count_comp = count_comp + 1

            train_val.append(history.history['loss'][-1])

            print("Mean loss over epoch: ", train_val[-1])

            y_pred = model.predict(X_valid)

            test_val_epoch = np.mean(np.abs(y_pred-y_valid))

            test_val.append(test_val_epoch)

            print("Test loss: ", test_val[-1])

            train_curves['train_val'] = np.array(train_val)
            train_curves['test_val'] = np.array(test_val)
            train_curves['grad_norm'] = np.array(norm_val)

            with open(save_folder + "/train_curves.pickle", 'wb') as f:
                pickle.dump(train_curves, f)

            # Save model after every 10th epoch and make predictions
            if ((i + 1) % 10 == 0):
                model.save(checkpoint_path)
                print("Model saved in ", save_folder)

                file_predict = save_folder + '/predictions.h5'
                f = h5py.File(file_predict, 'w')
                f['predictions'] = y_pred
                f.close()
                print("Predictions saved in: ", save_folder)

                if not os.path.exists(save_folder + '/train'):
                    os.makedirs(save_folder + '/train')

                y_pred_train = model.predict(X_train)
                np.save(save_folder + '/train/train_predictions', y_pred_train)

        print("Model saved in: ", save_folder)
        return model

    def visualize_train(self,img=None,lab=None,pcount=None,save_folder=None,idx=None):

        # img is the ground truth image
        # lab is the target count map of img
        # pcount is the predicted count map

        img=np.squeeze(img)
        pcount =np.reshape(pcount,[pcount.shape[1],-1,2])
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

        ax4.set_title("Predicted counts" )
        temp = []
        for i in range(pcount.shape[2]):
            temp.append(pcount[:, :, i])
        ax4.imshow(np.hstack(temp), interpolation='none')
        ax4.set_xticks([])
        ax4.set_yticks([])
        del temp

        pred_est = (np.sum(pcount,axis=(0,1))/self.ef).astype(np.int)
        lab_est = (np.sum(lab,axis=(0,1))/self.ef).astype(np.int)
        noutputs=lab.shape[2]

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
        plt.savefig(save_path+'comp_'+str(idx)+'.png',dpi=164)

    def predict(self,X_test,save_folder='cc_model_results'):
        checkpoint_path=save_folder+'/checkpoint/cc_model.ckpt'

        model = self.model
        model.load_weights(checkpoint_path)

        y_pred=model.predict(X_test)

        file_predict = save_folder + '/predictions.h5'
        f = h5py.File(file_predict, 'w')
        f['predictions'] = y_pred
        f.close()
        print("Predictions saved in: ", save_folder)
        return y_pred


'''
model = Model(inputs=[self.main_input], outputs=self.main_output)
opt = tf.keras.optimizers.Adam(lr=self.lr)
mae_loss = tf.keras.losses.mean_absolute_error()
model.compile(loss=mae_loss, optimizer=opt, metrics=['mae'])

return model
'''

'''
self.model.load_weights(checkpoint_path)
y_pred=self.model.predict(X_test)
'''

'''
model = Model(inputs=[self.main_input], outputs=self.main_output)
opt = tf.keras.optimizers.Adam(lr=self.lr)
mae_loss = tf.keras.losses.mean_absolute_error
model.compile(loss=self.custom_mae_loss, optimizer=opt)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, period=10)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=save_folder, write_graph=True)

for i in range(self.epochs):
    print("Epoch: {}".format(i+1))
    model.fit(x=X_train, y=y_train, batch_size=self.batch_sz,
          callbacks=[cp_callback, tb_callback])
    # Visulaize the training after every epoch

    img_predict = model.predict(img_valid.reshape([1, input_shape[0], input_shape[1], -1]))
    img_predict = np.squeeze(img_predict)[:, :, 1]
    self.visualize_train(img=img_valid, lab=count_valid, pcount=img_predict)

print("Model saved in: ", save_folder)
'''

'''
fig.canvas.draw()

img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# img is rgb, convert to opencv's default bgr
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Comparison",img)
cv2.waitKey(16)
'''