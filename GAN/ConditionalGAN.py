import os,path,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import math,glob,random,time

class CGAN:
    def __init__(self,isTraining,imageSize,labelSize,args):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = imageSize
        self.saveFolder = args.saveFolder
        self.reload = args.reload
        self.labelSize = labelSize
        self.buildModel()

        return

    def _fc_variable(self, weight_shape,name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _deconv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            output_channels = int(weight_shape[2])
            input_channels  = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape    , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def _deconv2d(self, x, W, output_shape, stride=1):
        # x           : [nBatch, height, width, in_channels]
        # output_shape: [nBatch, height, width, out_channels]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x) 

    def calcImageSize(self,dh,dw,stride):
        return int(math.ceil(float(dh)/float(stride))),int(math.ceil(float(dw)/float(stride)))

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

    def buildGenerator(self,z,label,reuse=False,isTraining=True):
        dim_0_h,dim_0_w = self.imageSize[0],self.imageSize[1]
        dim_1_h,dim_1_w = self.calcImageSize(dim_0_h, dim_0_w, stride=2)
        dim_2_h,dim_2_w = self.calcImageSize(dim_1_h, dim_1_w, stride=2)
        dim_3_h,dim_3_w = self.calcImageSize(dim_2_h, dim_2_w, stride=2)

        with tf.variable_scope("Generator") as scope:
            if reuse: scope.reuse_variables()

            l = tf.one_hot(label,self.labelSize,name="label_onehot")

            h = tf.concat([z,l],axis=1,name="concat_z")

            # fc1
            self.g_fc1_w, self.g_fc1_b = self._fc_variable([self.zdim+self.labelSize,256*dim_3_h*dim_3_w],name="fc1")
            h = tf.matmul(h, self.g_fc1_w) + self.g_fc1_b
            h = tf.nn.relu(h)

            #
            h = tf.reshape(h,(self.nBatch,dim_3_h,dim_3_h,256))

            # deconv3
            self.g_deconv3_w, self.g_deconv3_b = self._deconv_variable([5,5,256,128],name="deconv3")
            h = self._deconv2d(h,self.g_deconv3_w, output_shape=[self.nBatch,dim_2_h,dim_2_w,128], stride=2) + self.g_deconv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm3")
            h = tf.nn.relu(h)

            # deconv2
            self.g_deconv2_w, self.g_deconv2_b = self._deconv_variable([5,5,128,64],name="deconv2")
            h = self._deconv2d(h,self.g_deconv2_w, output_shape=[self.nBatch,dim_1_h,dim_1_w,64], stride=2) + self.g_deconv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm2")
            h = tf.nn.relu(h)

            # deconv1
            self.g_deconv1_w, self.g_deconv1_b = self._deconv_variable([5,5,64,3],name="deconv1")
            h = self._deconv2d(h,self.g_deconv1_w, output_shape=[self.nBatch,dim_0_h,dim_0_w,3], stride=2) + self.g_deconv1_b

            # sigmoid
            y = tf.tanh(h)

            ### summary
            if reuse:
                tf.summary.histogram("g_fc1_w"   ,self.g_fc1_w)
                tf.summary.histogram("g_fc1_b"   ,self.g_fc1_b)
                tf.summary.histogram("g_deconv1_w"   ,self.g_deconv1_w)
                tf.summary.histogram("g_deconv1_b"   ,self.g_deconv1_b)
                tf.summary.histogram("g_deconv2_w"   ,self.g_deconv2_w)
                tf.summary.histogram("g_deconv2_b"   ,self.g_deconv2_b)
                tf.summary.histogram("g_deconv3_w"   ,self.g_deconv3_w)
                tf.summary.histogram("g_deconv3_b"   ,self.g_deconv3_b)

        return y

    def buildDiscriminator(self,y,label,reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse: scope.reuse_variables()

            # conditional layer
            l = tf.one_hot(label,self.labelSize,name="label_onehot")
            l = tf.reshape(l,[self.nBatch,1,1,self.labelSize])
            k = tf.ones([self.nBatch,self.imageSize[0],self.imageSize[1],self.labelSize])
            k = k * l
            h = tf.concat([y,k],axis=3)

            # conv1
            self.d_conv1_w, self.d_conv1_b = self._conv_variable([5,5,3+self.labelSize,64],name="conv1")
            h = self._conv2d(h,self.d_conv1_w, stride=2) + self.d_conv1_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm1")
            h = self.leakyReLU(h)

            # conv2
            self.d_conv2_w, self.d_conv2_b = self._conv_variable([5,5,64,128],name="conv2")
            h = self._conv2d(h,self.d_conv2_w, stride=2) + self.d_conv2_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm2")
            h = self.leakyReLU(h)

            # conv3
            self.d_conv3_w, self.d_conv3_b = self._conv_variable([5,5,128,256],name="conv3")
            h = self._conv2d(h,self.d_conv3_w, stride=2) + self.d_conv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=self.isTraining, scope="dNorm3")
            h = self.leakyReLU(h)

            # fc1
            n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
            h = tf.reshape(h,[self.nBatch,n_h*n_w*n_f])
            self.d_fc1_w, self.d_fc1_b = self._fc_variable([n_h*n_w*n_f,1],name="fc1")
            h = tf.matmul(h, self.d_fc1_w) + self.d_fc1_b

            ### summary
            if not reuse:
                tf.summary.histogram("d_fc1_w"   ,self.d_fc1_w)
                tf.summary.histogram("d_fc1_b"   ,self.d_fc1_b)
                tf.summary.histogram("d_conv1_w"   ,self.d_conv1_w)
                tf.summary.histogram("d_conv1_b"   ,self.d_conv1_b)
                tf.summary.histogram("d_conv2_w"   ,self.d_conv2_w)
                tf.summary.histogram("d_conv2_b"   ,self.d_conv2_b)
                tf.summary.histogram("d_conv3_w"   ,self.d_conv3_w)
                tf.summary.histogram("d_conv3_b"   ,self.d_conv3_b)

        return h

    def buildModel(self):
        # define variables
        self.z      = tf.placeholder(tf.float32, [self.nBatch, self.zdim],name="z")
        self.l      = tf.placeholder(tf.int32  , [self.nBatch],name="label")

        self.y_real = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 3],name="image")

        self.y_fake = self.buildGenerator(self.z,self.l)
        self.y_sample = self.buildGenerator(self.z,self.l,reuse=True,isTraining=False)

        self.d_real  = self.buildDiscriminator(self.y_real,self.l)
        self.d_fake  = self.buildDiscriminator(self.y_fake,self.l,reuse=True)

        # define loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real,labels=tf.ones_like (self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,labels=tf.zeros_like(self.d_fake)))
        self.g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake,labels=tf.ones_like (self.d_fake)))
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        # define optimizer
        self.g_optimizer = tf.train.AdamOptimizer(self.learnRate,beta1=0.5).minimize(self.g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learnRate,beta1=0.5).minimize(self.d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

        ### summary
        tf.summary.scalar("d_loss_real"   ,self.d_loss_real)
        tf.summary.scalar("d_loss_fake"   ,self.d_loss_fake)
        tf.summary.scalar("d_loss"      ,self.d_loss)
        tf.summary.scalar("g_loss"      ,self.g_loss)

        #############################
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.15))
        self.sess = tf.Session(config=config)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)

        return