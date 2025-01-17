from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import h5py
import numpy as np
import os
import sys
import cv2
sys.path.append('./model/Discriminator/')
sys.path.append('./model/Generator')
sys.path.append('./model/Utilities/')
import Res_Gen
import HisDis
import PatchGAN70
import PatchGAN142
import MultiPatch
#import Stack_Gen
#import Sa_Gen
import PatchGAN34
import Utilities
import pickle
import matplotlib.pyplot as plt

class Model:
    """
    ToDo
    -) save()      - Save the current model parameter
    -) create()    - Create the model layers
    -) init()      - Initialize the model (load model if exists)
    -) load()      - Load the parameters from the file
    -) ToDo
    
    Only the following functions should be called from outside:
    -) ToDo
    -) constructor
    """
    
    def __init__(self,mod_name,data_file,valid_file,buffer_size=32,architecture='Res6',lambda_h=10.,lambda_c=10.,dis_noise=0.25,\
                 deconv='transpose',attention=0,patchgan='Patch70',save_folder = 'Models/',verbose=False,gen_only=False,data_file_B=None):
        """
        Create a Model (init). It will check, if a model with such a name has already been saved. If so, the model is being 
        loaded. Otherwise, a new model with this name will be created. It will only be saved, if the save function is being 
        called. The describtion of every parameter is given in the code below.
        
        INPUT: mod_name      - This is the name of the model. It is mainly used to establish the place, where the model is being 
                               saved.
               data_file     - hdf5 file that contains the dataset
               imsize        - The dimension of the input images
                              
        OUTPUT:             - The model
        """
        
        self.mod_name = mod_name                               # Model name (see above)
        self.save_folder = save_folder
        self.data_file = data_file                              # hdf5 data file
        self.valid_file = valid_file                            # hdf5 validation file
        self.data_file_B = data_file_B

        with h5py.File(self.data_file,"r") as f:
            self.a_chan = int(f['A/data'].shape[3])      # Number channels in A
            self.imsize = int(f['A/data'].shape[1])  # Image size (square image)
            self.a_size = int(f['A/data'].shape[0])  # Number of samples in A
            if not data_file_B:
                self.b_chan = int(f['B/data'].shape[3])  # Number channels in B
                self.b_size = int(f['B/data'].shape[0])  # Number of samples in B
            else:
                with h5py.File(self.data_file_B,'r') as file_b:
                    self.b_chan = int(file_b['B/data'].shape[3])      # Number channels in B
                    self.b_size = int(file_b['B/data'].shape[0])      # Number of samples in B
                
        # Reset all current saved tf stuff
        tf.reset_default_graph()
        
        self.architecture = architecture
        self.lambda_h = lambda_h
        self.lambda_c = lambda_c
        self.dis_noise_0 = dis_noise # ATTENTION: Name change from dis_noise to dis_noise_0
        self.deconv = deconv
        self.attention_flag = not not attention
        self.patchgan = patchgan
        self.verbose = verbose
        self.gen_only = gen_only  # If true, only the generator are used (and loaded)
        
        # Create the model that is built out of two discriminators and a generator
        self.create()
        
        # Image buffer
        self.buffer_size = buffer_size
        self.temp_b_s = 0.
        self.buffer_real_a = np.zeros([self.buffer_size,self.imsize,self.imsize,self.a_chan])
        self.buffer_real_b = np.zeros([self.buffer_size,self.imsize,self.imsize,self.b_chan])
        self.buffer_fake_a = np.zeros([self.buffer_size,self.imsize,self.imsize,self.a_chan])
        self.buffer_fake_b = np.zeros([self.buffer_size,self.imsize,self.imsize,self.b_chan])
        
        # Create the model saver
        with self.graph.as_default():
            if not self.gen_only:
                self.saver = tf.train.Saver()
            else:
                self.saver = tf.train.Saver(var_list=self.list_gen)
    
    def create(self):
        """
        Create the model. ToDo
        """
        # Create a graph and add all layers
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define variable learning rate and dis_noise
            self.relative_lr    = tf.placeholder_with_default([1.],[1],name="relative_lr")
            self.relative_lr    = self.relative_lr[0]
            
            self.rel_dis_noise  = tf.placeholder_with_default([1.],[1],name="rel_dis_noise")
            self.rel_dis_noise  = self.rel_dis_noise[0]
            self.dis_noise      = self.rel_dis_noise * self.dis_noise_0
            
            
            # Create the generator and discriminator
            network_type = 'res'
            if self.architecture == 'Res6':
                gen_dim =    [64,128,256,256,256,256,256,256,256,128,64]
                kernel_size =[7,3,3,3,3,3,3,3,3,3,3,7]
                network_type = 'res'
            elif self.architecture == 'Res9':
                gen_dim=    [64,128,256,256,256,256,256,256,256,256,256,256,128,64]
                kernel_size=[7,3,3,3,3,3,3,3,3,3,3,3,3,3,7]
                network_type = 'res'
            elif self.architecture == 'Stack3':
                gen_dim=    [32,   64,128,256,   256,   128,64,32,   32 ]
                kernel_size=[7,    3,3,3,        3,     3,3,3,       7,7]
                network_type = 'stack'
            elif self.architecture == 'Sa6':
                gen_dim =    [32,64,128,128,128,128,128,128,128,64,32]
                kernel_size =[7,3,3,3,3,3,3,3,3,3,3,7]
                network_type = 'sa'
            else:
                print('Unknown architecture')
                return None

            if network_type == 'res':
                self.genA       = Res_Gen.ResGen('BtoA',self.a_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
                self.genB       = Res_Gen.ResGen('AtoB',self.b_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
                if self.attention_flag:
                    self.att        = Res_Gen.ResGen('att',1,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
            else:
                print("Network not found!")
                sys.exit(1)

            '''
            elif network_type == 'sa':    
                self.genA       = Sa_Gen.SaGen('BtoA',self.a_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
                self.genB       = Sa_Gen.SaGen('AtoB',self.b_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
                if self.attention_flag:
                    self.att        = Sa_Gen.ResGen('att',1,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
            
            
            else:
                self.genA       = Stack_Gen.ResGen('BtoA',self.a_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
                self.genB       = Stack_Gen.ResGen('AtoB',self.b_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
                if self.attention_flag:
                    self.att        = Stack_Gen.ResGen('att',1,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
            '''

            if self.patchgan == 'Patch34':
                self.disA       = PatchGAN34.PatchGAN34('A',noise=self.dis_noise)
                self.disB       = PatchGAN34.PatchGAN34('B',noise=self.dis_noise)
            elif self.patchgan == 'Patch70':
                self.disA       = PatchGAN70.PatchGAN70('A',noise=self.dis_noise)
                self.disB       = PatchGAN70.PatchGAN70('B',noise=self.dis_noise)
            elif self.patchgan == 'Patch142':
                self.disA       = PatchGAN142.PatchGAN142('A',noise=self.dis_noise)
                self.disB       = PatchGAN142.PatchGAN142('B',noise=self.dis_noise)
            elif self.patchgan == 'MultiPatch':
                self.disA       = MultiPatch.MultiPatch('A',noise=self.dis_noise)
                self.disB       = MultiPatch.MultiPatch('B',noise=self.dis_noise)
            else:
                print('Unknown Patch discriminator type')
                return None

            self.disA_His = HisDis.HisDis('A',noise=self.dis_noise,keep_prob=1.)
            self.disB_His = HisDis.HisDis('B',noise=self.dis_noise,keep_prob=1.)
        
            # Create a placeholder for the input data
            self.A = tf.placeholder(tf.float32,[None, None, None, self.a_chan],name="a")
            self.B = tf.placeholder(tf.float32,[None, None, None, self.b_chan],name="b")
            
            if self.verbose:
                print('Size A: ' +str(self.a_chan)) # Often 1 --> Real
                print('Size B: ' +str(self.b_chan)) # Often 3 --> Syn
            
            # Create cycleGAN
            if self.attention_flag:
                self.attention = self.att.create(self.A,False)
                
            
            self.fake_A = self.genA.create(self.B,False)
            if self.attention_flag:
                self.fake_B_bef  = self.genB.create(self.A,False)
                self.fake_B = self.fake_B_bef * self.attention + (1. - self.attention)
            else:
                self.fake_B = self.genB.create(self.A,False)

            # Define the histogram loss
            t_A = tf.transpose(tf.reshape(self.A,[-1, self.a_chan]),[1,0])
            t_B = tf.transpose(tf.reshape(self.B,[-1, self.b_chan]),[1,0])
            t_fake_A = tf.transpose(tf.reshape(self.fake_A,[-1, self.a_chan]),[1,0])
            t_fake_B = tf.transpose(tf.reshape(self.fake_B,[-1, self.b_chan]),[1,0])

            self.s_A,_ = tf.nn.top_k(t_A,tf.shape(t_A)[1])
            self.s_B,_ = tf.nn.top_k(t_B,tf.shape(t_B)[1])
            self.s_fake_A,_ = tf.nn.top_k(t_fake_A,tf.shape(t_fake_A)[1])
            self.s_fake_B,_ = tf.nn.top_k(t_fake_B,tf.shape(t_fake_B)[1])
            
            self.m_A = tf.reshape(tf.reduce_mean(tf.reshape(self.s_A,[self.a_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_B = tf.reshape(tf.reduce_mean(tf.reshape(self.s_B,[self.b_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_fake_A = tf.reshape(tf.reduce_mean(tf.reshape(self.s_fake_A,[self.a_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_fake_B = tf.reshape(tf.reduce_mean(tf.reshape(self.s_fake_B,[self.b_chan, self.imsize, -1]),axis=2),[1, -1])
            
            # Define generator loss functions
            self.lambda_c = tf.placeholder_with_default([self.lambda_c],[1],name="lambda_c")
            self.lambda_c = self.lambda_c[0]
            self.lambda_h = tf.placeholder_with_default([self.lambda_h],[1],name="lambda_h")
            self.lambda_h = self.lambda_h[0]
            
            self.dis_real_A  = self.disA.create(self.A,False)
            self.dis_real_Ah = self.disA_His.create(self.m_A,False)
            self.dis_real_B  = self.disB.create(self.B,False)
            self.dis_real_Bh = self.disB_His.create(self.m_B,False)
            self.dis_fake_A  = self.disA.create(self.fake_A,True)
            self.dis_fake_Ah = self.disA_His.create(self.m_fake_A,True)
            self.dis_fake_B  = self.disB.create(self.fake_B,True)
            self.dis_fake_Bh = self.disB_His.create(self.m_fake_B,True)
            
            if self.attention_flag:
                self.cyc_A       = self.genA.create(self.fake_B,True) * self.attention +\
                                   (1. - self.attention) * (tf.random_normal(tf.shape(self.A),0.,0.1) + self.A)
            else:
                self.cyc_A       = self.genA.create(self.fake_B,True)
            self.cyc_B       = self.genB.create(self.fake_A,True)
            
            
            # Define cycle loss (eq. 2)
            self.loss_cyc_A  = tf.reduce_mean(tf.abs(self.cyc_A-self.A))
            self.loss_cyc_B  = tf.reduce_mean(tf.abs(self.cyc_B-self.B))
            
            self.loss_cyc    = self.loss_cyc_A + self.loss_cyc_B
            
            # Define discriminator losses (eq. 1)
            self.loss_dis_A  = (tf.reduce_mean(tf.square(self.dis_real_A)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_A)))*0.5 +\
                               (tf.reduce_mean(tf.square(self.dis_real_Ah)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_Ah)))*0.5*self.lambda_h
                                
                               
            self.loss_dis_B  = (tf.reduce_mean(tf.square(self.dis_real_B)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_B)))*0.5 +\
                               (tf.reduce_mean(tf.square(self.dis_real_Bh)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_Bh)))*0.5*self.lambda_h
            
            self.loss_gen_A  = tf.reduce_mean(tf.square(self.dis_fake_A)) +\
                               self.lambda_h * tf.reduce_mean(tf.square(self.dis_fake_Ah)) +\
                               self.lambda_c * self.loss_cyc/2.
            self.loss_gen_B  = tf.reduce_mean(tf.square(self.dis_fake_B)) +\
                               self.lambda_h * tf.reduce_mean(tf.square(self.dis_fake_Bh)) +\
                               self.lambda_c * self.loss_cyc/2.
                
        # Create the different optimizer
        with self.graph.as_default():
            # Optimizer for Gen
            self.list_gen        = []
            for var in tf.trainable_variables():
                if 'gen' in str(var):
                    self.list_gen.append(var)
            optimizer_gen   = tf.train.AdamOptimizer(learning_rate=self.relative_lr*0.0002,beta1=0.5)
            self.opt_gen    = optimizer_gen.minimize(self.loss_gen_A+self.loss_gen_B,var_list=self.list_gen)
            
            # Optimizer for Dis
            self.list_dis      = []
            for var in tf.trainable_variables():
                if 'dis' in str(var):
                    self.list_dis.append(var)
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.relative_lr*0.0002,beta1=0.5)
            self.opt_dis  = optimizer_dis.minimize(self.loss_dis_A + self.loss_dis_B,var_list=self.list_dis)
            
    def save(self,sess):
        """
        Save the model parameter in a ckpt file. The filename is as 
        follows:
        ./Models/<mod_name>.ckpt
        
        INPUT: sess         - The current running session
        """
        checkpoint_path = self.save_folder + 'checkpoint/'
        self.saver.save(sess,checkpoint_path + self.mod_name + ".ckpt")
            
    def init(self,sess):
        """
        Init the model. If the model exists in a file, load the model. Otherwise, initalize the variables
        INPUT: sess- The current running session
        """
        checkpoint_path = self.save_folder + 'checkpoint/'
        if not os.path.isfile(checkpoint_path + self.mod_name + ".ckpt.meta"):
            sess.run(tf.global_variables_initializer())
            return 0
        else:
            if self.gen_only:
                sess.run(tf.global_variables_initializer())
            self.load(sess)
            return 1
    
    def load(self,sess):
        """
        Load the model from the parameter file: ./Models/<mod_name>.ckpt
        INPUT: sess- The current running session
        """
        checkpoint_path = self.save_folder + 'checkpoint/'
        self.saver.restore(sess, checkpoint_path + self.mod_name + ".ckpt")

    def get_b_order(self,num_samples,idx_start = 0):
        if idx_start >= self.b_size:
            idx_start = idx_start - self.b_size
            idx_end = idx_start + num_samples
            if idx_end < self.b_size:
                b_order = np.arange(idx_start, idx_end)
            else:
                b_order = np.concatenate((np.arange(idx_start, self.b_size), np.arange(idx_end - self.b_size)))
                idx_end = idx_end - self.b_size

        else:
            idx_end = idx_start + num_samples
            if idx_end < self.b_size:
                b_order = np.arange(idx_start, idx_end)
            else:
                b_order = np.concatenate((np.arange(idx_start, self.b_size), np.arange(idx_end - self.b_size)))
                idx_end = idx_end - self.b_size

        assert len(b_order) == num_samples
        return [b_order,idx_end]

    
    def train(self,batch_size=32,lambda_c=0.,lambda_h=0.,n_epochs=1,save=True,syn_noise=0.,real_noise=0.,num_samples = None):

        f = h5py.File(self.data_file, 'r')
        if not num_samples:
            raise ValueError("Value for number of samples not fed!")


        # Fix the number of iterations: As many as required for a single pass over the data
        # An epoch iterates is a multiple of such iterations

        num_samples = min(self.a_size,self.b_size,num_samples)
        num_iterations = num_samples//batch_size
        a_order = np.random.permutation(self.a_size)[:num_samples]
        b_order = np.random.permutation(self.b_size)[:num_samples] #Used for Apr_27/


        if num_samples<batch_size:
            batch_size = num_samples
        
        if self.verbose:
            print('lambda_c: ' + str(lambda_c))
            print('lambda_h: ' + str(lambda_h))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if not os.path.exists(self.save_folder + 'Images/'):
            os.makedirs(self.save_folder + 'Images')

        with tf.Session(graph=self.graph, config = config) as sess:
            # initialize variables
            self.init(sess)

            rel_lr = 1.
            #idx_start = 0
            for curr_epoch in range(n_epochs):
                
                #[b_order, idx_start] = self.get_b_order(num_samples, idx_start)
                #b_order = np.random.choice(np.arange(self.b_size),num_samples,replace=False)
                np.random.shuffle(a_order)
                np.random.shuffle(b_order)

                if curr_epoch > 100:
                    rel_lr = 2. - curr_epoch/100.

                if curr_epoch < 50:
                    rel_noise = 0.9 ** curr_epoch
                else:
                    rel_noise = 0.


                with open(self.save_folder + 'log.txt','a+') as log:
                    log.write("\n=====================\n")
                    log.write("Epoch: {}".format(curr_epoch+1))


                vec_lcA = []
                vec_lcB = []

                vec_ldrA = []
                vec_ldrAh = []
                vec_ldrB = []
                vec_ldrBh = []
                vec_ldfA = []
                vec_ldfAh = []
                vec_ldfB = []
                vec_ldfBh = []

                vec_l_dis_A = []
                vec_l_dis_B = []
                vec_l_gen_A = []
                vec_l_gen_B = []

                for iteration in range(num_iterations):
                    images_a   = f['A/data'][np.sort(a_order[(iteration*batch_size):((iteration+1)*batch_size)]),:,:,:]
                    if not self.data_file_B:
                        images_b   = f['B/data'][np.sort(b_order[(iteration*batch_size):((iteration+1)*batch_size)]),:,:,:]
                    else:
                        with h5py.File(self.data_file_B,'r') as file_b:
                            images_b   = file_b['B/data'][np.sort(b_order[(iteration*batch_size):((iteration+1)*batch_size)]),:,:,:]

                    #Check whether images A and B are in correct format or not
                    if images_a.dtype=='uint8':
                        images_a=images_a/float(2**8-1)
                    elif images_a.dtype=='uint16':
                        images_a=images_a/float(2**16-1)
                    else:
                        raise ValueError('Dataset A is not int8 or int16')
                    if images_b.dtype=='uint8':
                        images_b=images_b/float(2**8-1)
                    elif images_b.dtype=='uint16':
                        images_b=images_b/float(2**16-1)
                    else:
                        raise ValueError('Dataset B is not int8 or int16')

                    assert np.all(np.max(images_a)<1.5)
                    assert np.all(np.max(images_b)<1.5)
                    assert np.all(np.min(images_a)>=0.)
                    assert np.all(np.min(images_b)>=0)

                    images_a  += np.random.randn(*images_a.shape)*real_noise
                    images_b  += np.random.randn(*images_b.shape)*syn_noise

                    if self.attention_flag:
                        feed_data = {self.A: images_a,self.B: images_b,self.lambda_c: lambda_c,self.lambda_h: lambda_h,\
                                     self.relative_lr: rel_lr,self.rel_dis_noise: rel_noise}

                        var_to_run = [self.opt_gen,self.loss_gen_A,self.fake_A,self.loss_gen_B,self.fake_B,\
                                      self.cyc_A,self.cyc_B,self.s_A,self.s_B,self.s_fake_A,self.s_fake_B,\
                                      self.attention,self.fake_B_bef,self.loss_cyc_A,self.loss_cyc_B]
                        _, l_gen_A, im_fake_A, l_gen_B, im_fake_B,\
                        cyc_A, cyc_B, sA, sB, sfA, sfB, \
                        attention_out, fakeB_pre_att,lcA,lcB = sess.run(var_to_run,feed_dict= feed_data)
                    else:
                        feed_data = {self.A: images_a,self.B: images_b,self.lambda_c: lambda_c,self.lambda_h: lambda_h,\
                                     self.relative_lr: rel_lr,self.rel_dis_noise: rel_noise}

                        var_to_run = [self.opt_gen,self.loss_gen_A,self.fake_A,self.loss_gen_B,self.fake_B,\
                                      self.cyc_A,self.cyc_B,self.s_A,self.s_B,self.s_fake_A,self.s_fake_B,\
                                      self.loss_cyc_A,self.loss_cyc_B]
                        _, l_gen_A, im_fake_A, l_gen_B, im_fake_B, \
                        cyc_A, cyc_B, sA, sB, sfA, sfB, lcA, lcB = sess.run(var_to_run,feed_dict=feed_data)

                    if self.temp_b_s >= self.buffer_size:
                        rand_vec_a = np.random.permutation(self.buffer_size)[:batch_size]
                        rand_vec_b = np.random.permutation(self.buffer_size)[:batch_size]

                        self.buffer_real_a[rand_vec_a,...] = images_a
                        self.buffer_real_b[rand_vec_b,...] = images_b
                        self.buffer_fake_a[rand_vec_a,...] = im_fake_A
                        self.buffer_fake_b[rand_vec_b,...] = im_fake_B
                    else:
                        low = int(self.temp_b_s)
                        high = int(min(self.temp_b_s + batch_size,self.buffer_size))
                        self.temp_b_s = high

                        self.buffer_real_a[low:high,...] = images_a[:(high-low),...]
                        self.buffer_real_b[low:high,...] = images_b[:(high-low),...]
                        self.buffer_fake_a[low:high,...] = im_fake_A[:(high-low),...]
                        self.buffer_fake_b[low:high,...] = im_fake_B[:(high-low),...]

                    # Create dataset out of buffer and gen images to train dis
                    dis_real_a = np.copy(images_a)
                    dis_real_b = np.copy(images_b)
                    dis_fake_a = np.copy(im_fake_A)
                    dis_fake_b = np.copy(im_fake_B)

                    half_b_s = int(batch_size/2)
                    rand_vec_a = np.random.permutation(self.temp_b_s)[:half_b_s]
                    rand_vec_b = np.random.permutation(self.temp_b_s)[:half_b_s]
                    dis_real_a[:half_b_s,...] =  self.buffer_real_a[rand_vec_a,...]
                    dis_fake_a[:half_b_s,...] =  self.buffer_fake_a[rand_vec_a,...]
                    dis_real_b[:half_b_s,...] =  self.buffer_real_b[rand_vec_b,...]
                    dis_fake_b[:half_b_s,...] =  self.buffer_fake_b[rand_vec_b,...]

                    var_to_run = [self.opt_dis,self.loss_dis_A,self.loss_dis_B,self.dis_real_A,self.dis_real_Ah,\
                                 self.dis_fake_A,self.dis_fake_Ah,self.dis_real_B,self.dis_real_Bh,self.dis_fake_B,self.dis_fake_Bh]

                    feed_data = {self.A: dis_real_a,self.B: dis_real_b,self.fake_A: dis_fake_a,self.fake_B: dis_fake_b,\
                                 self.lambda_c: lambda_c,self.lambda_h: lambda_h,self.relative_lr: rel_lr,self.rel_dis_noise: rel_noise}

                    _, l_dis_A, l_dis_B, \
                    ldrA,ldrAh,ldfA,ldfAh,\
                    ldrB,ldrBh,ldfB,ldfBh = sess.run(var_to_run,feed_dict = feed_data)

                    vec_l_dis_A.append(l_dis_A)
                    vec_l_dis_B.append(l_dis_B)
                    vec_l_gen_A.append(l_gen_A)
                    vec_l_gen_B.append(l_gen_B)

                    vec_lcA.append(lcA)
                    vec_lcB.append(lcB)

                    vec_ldrA.append(ldrA)
                    vec_ldrAh.append(ldrAh)
                    vec_ldrB.append(ldrB)
                    vec_ldrBh.append(ldrBh)
                    vec_ldfA.append(ldfA)
                    vec_ldfAh.append(ldfAh)
                    vec_ldfB.append(ldfB)
                    vec_ldfBh.append(ldfBh)

                    if np.shape(images_b)[-1]==4:
                        images_b=np.vstack((images_b[0,:,:,0:3],np.tile(images_b[0,:,:,3].reshape(320,320,1),[1,1,3])))
                        im_fake_B=np.vstack((im_fake_B[0,:,:,0:3],np.tile(im_fake_B[0,:,:,3].reshape(320,320,1),[1,1,3])))
                        cyc_B=np.vstack((cyc_B[0,:,:,0:3],np.tile(cyc_B[0,:,:,3].reshape(320,320,1),[1,1,3])))
                        images_b=images_b[np.newaxis,:,:,:]
                        im_fake_B=im_fake_B[np.newaxis,:,:,:]
                        cyc_B=cyc_B[np.newaxis,:,:,:]

                #Output the training losses every epoch
                with open(self.save_folder + 'log.txt','a+') as log:
                    log.write("\nTrain: {}/{} ({:.1f}%)".format(curr_epoch+1, n_epochs,(curr_epoch+1) * 100 / (n_epochs)) + \
                          "          Loss_dis_A={:.4f},   Loss_dis_B={:.4f}".format(np.mean(vec_l_dis_A),np.mean(vec_l_dis_B)) + \
                          ",   Loss_gen_A={:.4f},   Loss_gen_B={:.4f}".format(np.mean(vec_l_gen_A),np.mean(vec_l_gen_B)))

                #Save the losses every epoch
                losses = [np.mean(vec_l_dis_A),np.mean(vec_l_dis_B),np.mean(vec_l_gen_A),np.mean(vec_l_gen_B)]

                if np.all(np.isnan(losses[0])):
                    with open(self.save_folder + 'log.txt','a+') as log:
                        log.write("\nNans encountered. Exiting ...")
                        sys.exit(1)

                pf_dA = [np.mean(np.square(np.array(vec_ldfA))), np.mean(np.square(np.array(vec_ldfAh))),
                                  np.mean(np.array(lcA))]
                pf_dB = [np.mean(np.square(np.array(vec_ldfB))), np.mean(np.square(np.array(vec_ldfBh))),
                                  np.mean(np.array(lcB))]
                pr_dA = [np.mean(np.square(np.array(vec_ldrA))), np.mean(np.square(1. - np.array(vec_ldfA))), \
                                  np.mean(np.square(np.array(vec_ldrAh))), np.mean(np.square(1. - np.array(vec_ldfAh)))]
                pr_dB = [np.mean(np.square(np.array(vec_ldrB))), np.mean(np.square(1. - np.array(vec_ldfB))), \
                                  np.mean(np.square(np.array(vec_ldrBh))), np.mean(np.square(1. - np.array(vec_ldfBh)))]

                self.save_losses(losses,pf_dA,pf_dB,pr_dA,pr_dB)

                '''
                if self.attention_flag:
                    sneak_peak = Utilities.produce_tiled_images(images_a, images_b, im_fake_A, im_fake_B, cyc_A,
                                                                cyc_B, attention_out, fakeB_pre_att)
                else:
                    sneak_peak = Utilities.produce_tiled_images(images_a, images_b, im_fake_A, im_fake_B, cyc_A,
                                                                cyc_B)

                cv2.imwrite(self.save_folder + "Images/" + self.mod_name + "_Epoch_" + str(curr_epoch+1) + ".png",
                            sneak_peak[:, :, [2, 1, 0]] * 255)
                '''

                if (num_samples==1700):
                    # Visualize training every epoch

                    with h5py.File(self.valid_file, 'r') as valid_file:
                        im = np.array(valid_file['raw/data'][0, :, :, 0])
                        gt = np.array(valid_file['gt_SI/data'][0, :, :, 0])

                    pred_im = sess.run(self.fake_B,feed_dict={self.A: im[None,:,:,None],self.lambda_c:lambda_c,\
                                                    self.lambda_h:lambda_h})
                    pred_im = np.squeeze(np.minimum(np.maximum(pred_im,0),1))
                    fig_save_path = self.save_folder + "Images/" + self.mod_name + "_Epoch_" + str(curr_epoch+1) + ".png"
                    Utilities.visualize_train(im,gt,pred_im,path = fig_save_path)
                
                # Save model every 5 epochs
                if (save and (curr_epoch+1)%5==0 or curr_epoch == n_epochs-1):
                    self.save(sess)
        f.close()

    def save_losses(self,losses,pf_dA,pf_dB,pr_dA,pr_dB):
        # loss strcuture: mean losses over iteration wise stacked over epochs
        # loss structure: [epochs,4]

        if os.path.isfile(self.save_folder + 'train_losses.pickle'):
            with open(self.save_folder + 'train_losses.pickle','rb') as file:
                train_losses = pickle.load(file,encoding="bytes")

            all_losses = train_losses['all_losses']
            fake_A = train_losses['fake_A']
            fake_B = train_losses['fake_B']
            real_A = train_losses['real_A']
            real_B = train_losses['real_B']

            all_losses = np.concatenate((all_losses, np.array(losses)[None, :]), axis=0)
            fake_A = np.concatenate((fake_A,np.array(pf_dA)[None,:]),axis=0)
            fake_B = np.concatenate((fake_B, np.array(pf_dB)[None,:]), axis=0)
            real_A = np.concatenate((real_A, np.array(pr_dA)[None,:]), axis=0)
            real_B = np.concatenate((real_B, np.array(pr_dB)[None,:]), axis=0)
        else:
            all_losses = np.array(losses)[None,:]
            fake_A = np.array(pf_dA)[None,:]
            fake_B = np.array(pf_dB)[None,:]
            real_A = np.array(pr_dA)[None,:]
            real_B = np.array(pr_dB)[None,:]

        train_losses = {'all_losses':all_losses,'fake_A':fake_A,'fake_B':fake_B,\
                        'real_A':real_A,'real_B':real_B}

        with open(self.save_folder + 'train_losses.pickle','wb') as file:
            pickle.dump(train_losses,file)

    def data_gen_B(self,data_file,batch_size=32,lambda_c=0.,lambda_h=0.):

        f = h5py.File(data_file, "r")

        # Find number of samples
        images = np.array(f['raw/data']) # Data in float64 format, already normalized from [0-1]
        im_sz = images.shape[1:3]
        num_samples = images.shape[0]
        if num_samples%batch_size==0:
            num_iterations = num_samples // batch_size
        else:
            num_iterations = num_samples//batch_size + 1
        pred_channels = self.b_chan

        gen_data = np.zeros((num_samples, im_sz[0], im_sz[1], pred_channels),dtype=np.uint16)

        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)

            for iteration in range(num_iterations):
                images_a = images[(iteration * batch_size):((iteration + 1) * batch_size), :, :, :]
                gen_B = sess.run(self.fake_B, feed_dict={self.A: images_a, \
                                                         self.lambda_c: lambda_c, \
                                                         self.lambda_h: lambda_h})
                gen_data[(iteration * batch_size):((iteration + 1) * batch_size), :, :, :] = (
                            np.minimum(np.maximum(gen_B, 0), 1) * (2 ** 16 - 1)).astype(np.uint16)

        with open(self.save_folder + 'log.txt', 'a+') as log:
            log.write("\nGenerated B for given data ...")

        with h5py.File(self.save_folder + self.mod_name + '_gen_B.h5', "w") as f_save:
            group = f_save.create_group('B')
            group.create_dataset(name='data', data=gen_data, dtype=np.uint16)

        f.close()

    def predict(self,lambda_c=0.,lambda_h=0.):
        f = h5py.File(self.data_file,"r")
        
        rand_a = np.random.randint(self.a_size-32)
        rand_b = np.random.randint(self.b_size-32)
        
        images_a = f['A/data'][rand_a:(rand_a+32),:,:,:]/255.
        images_b = f['B/data'][rand_b:(rand_b+32),:,:,:]/255.
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
                
            fake_A, fake_B, cyc_A, cyc_B = \
                sess.run([self.fake_A,self.fake_B,self.cyc_A,self.cyc_B],\
                         feed_dict={self.A: images_a,\
                                    self.B: images_b,\
                                    self.lambda_c: lambda_c,\
                                    self.lambda_h: lambda_h})
            
        f.close()
        return images_a, images_b, fake_A, fake_B, cyc_A, cyc_B


    
    def generator_A(self,batch_size=32,lambda_c=0.,lambda_h=0.):
        f = h5py.File(self.data_file,"r")
        f_save = h5py.File(self.save_folder + self.mod_name + '_gen_A.h5',"w")
        
        # Find number of samples
        num_samples    = self.b_size
        num_iterations = num_samples // batch_size
                
        gen_data       = np.zeros((f['B/data'].shape[0],f['B/data'].shape[1],f['B/data'].shape[2],f['A/data'].shape[3]),dtype=np.uint16)
        
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
            
            for iteration in range(num_iterations):    
                images_b   = f['B/data'][(iteration*batch_size):((iteration+1)*batch_size),:,:,:]
                if images_b.dtype=='uint8':
                    images_b=images_b/float(2**8-1)
                elif images_b.dtype=='uint16':
                    images_b=images_b/float(2**16-1)
                else:
                    raise ValueError('Dataset B is not int8 or int16')

                gen_A = sess.run(self.fake_A,feed_dict={self.B: images_b,\
                                                        self.lambda_c: lambda_c,\
                                                        self.lambda_h: lambda_h})
                gen_data[(iteration*batch_size):((iteration+1)*batch_size),:,:,:] = (np.minimum(np.maximum(gen_A,0),1)*(2**16-1)).astype(np.uint16)

                with open(self.save_folder + 'log.txt','a+') as log:
                    log.write("\nGenerator A: {}/{} ({:.1f}%)".format(iteration+1, num_iterations, iteration*100/(num_iterations-1)))
        
        group = f_save.create_group('A')
        group.create_dataset(name='data', data=gen_data,dtype=np.uint16)
        
        f_save.close()
        f.close()
        
        return None

    def generator_B(self,batch_size=32,lambda_c=0.,lambda_h=0.,checkpoint_path=None):
        f = h5py.File(self.data_file,"r")
        f_save = h5py.File(self.save_folder + self.mod_name + '_gen_B.h5',"w")

        # Find number of samples
        num_samples    = self.a_size
        if num_samples%batch_size==0:
            num_iterations = num_samples // batch_size
        else:
            num_iterations = num_samples//batch_size + 1

        gen_data       = np.zeros((f['A/data'].shape[0],f['A/data'].shape[1],f['A/data'].shape[2],self.b_chan),dtype=np.uint16)

        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            if not checkpoint_path:
                self.init(sess)
            else:
                if not os.path.isfile(checkpoint_path + self.mod_name + ".ckpt.meta"):
                    print("Checkpoint not found. Exiting")
                    sys.exit(1)
                self.saver.restore(sess, checkpoint_path + self.mod_name + ".ckpt")

            for iteration in range(num_iterations):
                images_a   = f['A/data'][(iteration*batch_size):((iteration+1)*batch_size),:,:,:]
                if images_a.dtype=='uint8':
                    images_a=images_a/float(2**8-1)
                elif images_a.dtype=='uint16':
                    images_a=images_a/float(2**16-1)
                else:
                    raise ValueError('Dataset A is not int8 or int16')

                gen_B = sess.run(self.fake_B,feed_dict={self.A: images_a,\
                                                        self.lambda_c: lambda_c,\
                                                        self.lambda_h: lambda_h})
                gen_data[(iteration*batch_size):((iteration+1)*batch_size),:,:,:] = (np.minimum(np.maximum(gen_B,0),1)*(2**16-1)).astype(np.uint16)

                with open(self.save_folder + 'log.txt','a+') as log:
                    log.write("\nGenerator B: {}/{} ({:.1f}%)".format(iteration+1, num_iterations, iteration*100/(num_iterations-1)))

        group = f_save.create_group('B')
        group.create_dataset(name='data', data=gen_data,dtype=np.uint16)

        f_save.close()
        f.close()

        return None

    def get_loss(self,lambda_c=0.,lambda_h=0.):
        f = h5py.File(self.data_file,"r")
        
        rand_a = np.random.randint(self.a_size-32)
        rand_b = np.random.randint(self.b_size-32)
        
        images_a   = f['A/data'][rand_a:(rand_a+32),:,:,:]/255.
        images_b   = f['B/data'][rand_b:(rand_b+32),:,:,:]/255.
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
                
            l_rA,l_rB,l_fA,l_fB = \
                sess.run([self.dis_real_A,self.dis_real_B,self.dis_fake_A,self.dis_fake_B,],\
                         feed_dict={self.A: images_a,\
                                    self.B: images_b,\
                                    self.lambda_c: lambda_c,\
                                    self.lambda_h: lambda_h})
            
        f.close()
        return l_rA,l_rB,l_fA,l_fB
    
    def save_graph(self):
        tf.summary.FileWriter(self.save_folder + self.mod_name + ".logs",graph=self.graph)

    def print_gradients(self):
        """
        This function tests if the gradients exist and if they do, if the variable is being trained.
        0    Gradient exists but is 0. This is probably bad
        1    Gradient exists and is bigger than 0. It is being trained
        2    Gradient does not exist and is not supposed to be trained
        3    Gradient exists and is bigger than 0. It is not being trained
        4    Gradient does not exist but it is supposed to be trained. This must not happen
        """
        f              = h5py.File(self.data_file,"r")
        
        images_a   = f['A/data'][0:1,:,:,:]
        images_b   = f['B/data'][0:1,:,:,:]
        
        with tf.Session(graph=self.graph) as sess:
            self.init(sess)
            
            train_var = tf.trainable_variables()
            losses    = [self.loss_dis_A, self.loss_dis_B, self.loss_gen_A, self.loss_gen_B, self.loss_gen_A + self.loss_gen_B]
            loss_name = ["dis_A", "dis_B", "gen_A", "gen_B", "genAB"]
            
            length = 0
            for i in range(len(train_var)):
                length = max(length,len(tf.trainable_variables()[i].name))
            
            length += 3
            
            non_zero  = np.zeros((len(train_var),len(losses)),dtype=np.int)
            
            for i in range(len(train_var)):
                for j in range(len(losses)):
                    print(i,j,train_var[i].name,loss_name[j],end='    ')
                    try:
                        grad = sess.run(tf.gradients(ys=losses[j],xs=train_var[i]),\
                             feed_dict={self.A: images_a,\
                                        self.B: images_b})
                        if np.sum(np.abs(grad)) > 0:
                            if j == 0 and train_var[i] in self.list_dis_A:
                                non_zero[i,j] = 1.
                                print(1)
                            elif j == 1 and train_var[i] in self.list_dis_B:
                                non_zero[i,j] = 1.
                                print(1)
                            elif j >= 2 and train_var[i] in self.list_gen:
                                non_zero[i,j] = 1.
                                print(1)
                            else:
                                non_zero[i,j] = 3.
                                print(3)
                        else:
                            print(0)
                    except:
                        if j == 0 and train_var[i] in self.list_dis_A:
                            non_zero[i,j] = 4.
                            print(4)
                        elif j == 1 and train_var[i] in self.list_dis_B:
                            non_zero[i,j] = 4.
                            print(4)
                        elif j >= 2 and train_var[i] in self.list_gen:
                            non_zero[i,j] = 4.
                            print(4)
                        else:
                            non_zero[i,j] = 2.
                            print(2)
            # Print the header line
            for i in range(length):
                print(' ',end='')
            for j in range(len(loss_name)):
                print(loss_name[j],end='  ')
            print(' ')
            for i in range(len(train_var)):
                print(tf.trainable_variables()[i].name,end='  ')
                for j in range(len(tf.trainable_variables()[i].name),length):
                    print(' ',end='')
                for j in range(len(losses)):
                    print(str(non_zero[i,j]),end='      ')
                print('')
        f.close()
    
    def print_count_variables(self):
        with tf.Session(graph=self.graph):
            count = 0
            for var in tf.trainable_variables():
                count = count + int(np.prod(var.shape))
            print('Total number of trainable variables in model: ' + str(count))
            
        with tf.Session(graph=self.graph):
            count = 0
            for var in tf.trainable_variables():
                if 'dis_A' in str(var):
                    count = count + int(np.prod(var.shape))
            print('Total number of trainable variables in dis_A:  ' + str(count))
            
        with tf.Session(graph=self.graph):
            count = 0
            for var in tf.trainable_variables():
                if 'dis_B' in str(var):
                    count = count + int(np.prod(var.shape))
            print('Total number of trainable variables in dis_B:  ' + str(count))
            
        with tf.Session(graph=self.graph):
            count = 0
            for var in tf.trainable_variables():
                if 'gen_BtoA' in str(var):
                    count = count + int(np.prod(var.shape))
            print('Total number of trainable variables in gen_A:  ' + str(count))
            
        with tf.Session(graph=self.graph):
            count = 0
            for var in tf.trainable_variables():
                if 'gen_AtoB' in str(var):
                    count = count + int(np.prod(var.shape))
            print('Total number of trainable variables in gen_B:  ' + str(count))
    
    def print_train_and_not_train_variables(self):
        with tf.Session(graph=self.graph):
            count = 0
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                count = count + int(np.prod(var.shape))
            print('Total number of variables (trainable + not trainable): ' + str(count))