import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.models import load_model
import os
import h5py
import numpy as np


class cc_model():
    def __init__(self, input_shape, patch_size, lr=0.005, batch_sz=5, epochs=100, scope='cc_model'):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.main_input = Input(shape=input_shape, name='main_input')
        self.main_output = []
        self.lr = lr

        with tf.variable_scope(scope):
            self.model = self.create_model()

    def getModel(self):
        return self.model

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

    def create_model(self):
        net = self.ConvFactory1(self.main_input, num_filters=64, pad=self.patch_size, filter_size=3)
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
        model.compile(loss=mae_loss, optimizer=opt, metrics=['mae'])

        return model

    def train_model(self, X_train, y_train, save_folder="cc_model_results"):
        checkpoint_path = save_folder + "/checkpoint/cc_model.ckpt"

        model = self.model
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0, period=10)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=save_folder, write_graph=True)

        model.fit(x=X_train, y=y_train, batch_size=self.batch_sz, epochs=self.epochs,
                  callbacks=[cp_callback, tb_callback])

        print("Model saved in: ", save_folder)
        return model

    def predict(self, X_test, save_folder='cc_model_results'):
        checkpoint_path = save_folder + '/checkpoint/cc_model.ckpt'
        self.model.load_weights(checkpoint_path)
        y_pred = self.model.predict(X_test)
        file_predict = save_folder + '/predictions.h5'
        f = h5py.File(file_predict, 'w')
        f['predictions'] = np.squeeze(y_pred)
        f.close()
        print("Predictions saved in: ", save_folder)
        return y_pred