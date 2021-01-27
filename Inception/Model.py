# -*- coding: utf-8 -*-
"""

Created on Sat Mar  2 15:47:25 2019

File:Model.py

@author: WZX

"""

import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from data_generator import dataset_split,generator
from sklearn.metrics import confusion_matrix





class Model(object):
    """Inception-v4 implementation"""

    def __init__(self,inputShape,outputShape,ckptPath = 'trainModel/model',modelPath = 'saveModel/model'):
        """

        Defines the input and output variables and stores the save locations of the model



        Keyword arguments:

        input_shape -- The shape of the input tensor, indexed by [sample, row, col, channel]

        output_shape -- The shape of the output tensor, indexed by [sample, class]

        ckpt_path -- The save path for checkpoints during training

        model_path -- The save path for the forward propagation subgraph. The generated model

                      can be used to classify images without allocating space for the gradients

        """
        self.ckptPath = ckptPath          #保存训练进度
        self.modelPath = modelPath        #保存模型参数
        self.numclass = outputShape[1]    #类别数
        self.x = tf.placeholder(tf.float32,inputShape)   #输入口
        self.y = tf.placeholder(tf.float32,outputShape)  #标签输入口


        self.inception_v4()  #运行网络



    def print_activations(self,layer):
        """打印各层输出大小"""

        print(layer.op.name,'',layer.get_shape().as_list())



    def inception_v4(self):
        '''

        Define inception_v4 net

        '''

        def BN_layer(tensor,epsilon = 1e-3):
            batch_mean,batch_var = tf.nn.moments(tensor,[0])
            return tf.nn.batch_normalization(tensor,batch_mean,batch_var,offset = None,scale = None,variance_epsilon = epsilon)



        self.keep_prob = tf.placeholder(tf.float32)


        #Input

        with tf.variable_scope('Input'):
            X = self.x
            #X = tf.clip_by_value(self.x,1e-8,1.0)
            #self.print_activations(X)

        #Stem

        with tf.variable_scope('Stem'):
            stem = self.Stem(X)

        #Inception-A

        inception_a = {-1 : stem}
        for i in range(4):
            with tf.variable_scope('Inception-A_'+str(i+1)):
                inception_a[i] = self.inception_A(inception_a[i-1])
                #inception_a[i] = BN_layer(self.inception_A(inception_a[i-1]))


        #Reduction-A

        with tf.variable_scope('Reduction-A'):
            reduction_a = self.reduction_A(inception_a[3])
            #reduction_a = BN_layer(self.reduction_A(inception_a[0]))


        #Inception-B
        inception_b = {-1 : reduction_a}
        for i in range(7):
            with tf.variable_scope('Inception-B_'+str(i+1)):
                inception_b[i] = self.inception_B(inception_b[i-1])
                #inception_b[i] = BN_layer(self.inception_B(inception_b[i-1]))


        #Reduction-B
        with tf.variable_scope('Reduction-B'):
            reduction_b = self.reduction_B(inception_b[6])
            #reduction_b = BN_layer(self.reduction_B(inception_b[0]))


        #Inception-C
        inception_c = {-1 : reduction_b}
        for i in range(3):
            with tf.variable_scope('Inception-C_'+str(i+1)):
                inception_c[i] = self.inception_C(inception_c[i-1])
                #inception_c[i] = BN_layer(self.inception_C(inception_c[i-1]))

        #Average Pooling

        with tf.variable_scope('Average_Pooling'):
            pool = tf.nn.avg_pool(inception_c[2],[1,8,8,1],[1,1,1,1],padding = 'VALID')
            #self.print_activations(pool)

        self.pool_reshape = tf.reshape(pool,[-1,1536])
        #self.print_activations(pool_reshape)

        #Dropout

        self.pool_drop = tf.nn.dropout(self.pool_reshape,self.keep_prob,name = 'Dropout')
        #self.print_activations(pool_drop)


        #FC layer

        self.fc = self.dense(self.pool_drop,'fully_connect',self.numclass)


        #Softmax


        self.y_hat = tf.nn.softmax(self.fc, name = 'y_hat')
        #self.print_activations(self.y_hat)

        #Creates a saver object exclusively for the forward propagation subgraph


        model_variables = tf.get_collection_ref('tf.GraphKeys.MODEL_VARIABLES')
        self.model_saver = tf.train.Saver(model_variables)





    def Stem(self,tensor):
        '''

        Generates the graph for the stem subgraph of the Inception v4 model

        '''


        conv_1     = Model.conv(tensor,    'conv_1',     [3, 3, 3, 32],   [1, 2, 2, 1], padding='VALID')



        conv_2     = Model.conv(conv_1,    'conv_2',     [3, 3, 32, 32],  [1, 1, 1, 1], padding='VALID')



        conv_3     = Model.conv(conv_2,    'conv_3',     [3, 3, 32, 64],  [1, 1, 1, 1])





        pool_4_1   = tf.nn.max_pool(conv_3, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name = 'pool_4_1')


        conv_4_2   = Model.conv(conv_3,    'conv_4_2',   [3, 3, 64, 96],  [1, 2, 2, 1], padding='VALID')


        concat_1   = tf.concat([pool_4_1, conv_4_2], axis=3, name = 'concat_1')



        conv_5_1_1 = Model.conv(concat_1,  'conv_5_1_1', [1, 1, 160, 64], [1, 1, 1, 1])

        conv_5_1_2 = Model.conv(conv_5_1_1, 'conv_5_1_2', [3, 3, 64, 96],  [1, 1, 1, 1], padding='VALID')



        conv_5_2_1 = Model.conv(concat_1,   'conv_5_2_1', [1, 1, 160, 64], [1, 1, 1, 1])

        conv_5_2_2 = Model.conv(conv_5_2_1, 'conv_5_2_2', [7, 1, 64, 64],  [1, 1, 1, 1])

        conv_5_2_3 = Model.conv(conv_5_2_2, 'conv_5_2_3', [1, 7, 64, 64],  [1, 1, 1, 1])

        conv_5_2_4 = Model.conv(conv_5_2_3, 'conv_5_2_4', [3, 3, 64, 96],  [1, 1, 1, 1], padding='VALID')



        concat_2   = tf.concat([conv_5_1_2, conv_5_2_4], axis=3, name = 'concat_2')



        conv_6_1   = Model.conv(concat_2,   'conv_6_1_1', [3, 3, 192, 192],  [1, 2, 2, 1], padding='VALID')

        pool_6_2   = tf.nn.max_pool(concat_2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name = 'pool_6_2')



        concat_3   = tf.concat([conv_6_1, pool_6_2], axis=3, name = 'concat_3')


        #打印各层输出尺寸
        #self.print_activations(conv_1)
        #self.print_activations(conv_2)
        #self.print_activations(conv_3)
        #self.print_activations(concat_1)
        #self.print_activations(concat_2)
        #self.print_activations(concat_3)


        return concat_3



    def inception_A(self,tensor):

        '''

        Generates the graph for the Inception-A subgraph of the Inception v4 model

        '''

        pool_1_1 = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name = 'pool_1_1')

        conv_1_1 = Model.conv(pool_1_1, 'conv_1_1', [1, 1, 384, 96], [1, 1, 1, 1])



        conv_2_1 = Model.conv(tensor,  'conv_2_1', [1, 1, 384, 96], [1, 1, 1, 1])



        conv_3_1 = Model.conv(tensor,  'conv_3_1', [1, 1, 384, 64], [1, 1, 1, 1])

        conv_3_2 = Model.conv(conv_3_1, 'conv_3_2', [3, 3, 64, 96],  [1, 1, 1, 1])



        conv_4_1 = Model.conv(tensor,  'conv_4_1', [1, 1, 384, 64], [1, 1, 1, 1])

        conv_4_2 = Model.conv(conv_4_1, 'conv_4_2', [3, 3, 64, 96],  [1, 1, 1, 1])

        conv_4_3 = Model.conv(conv_4_2, 'conv_4_3', [3, 3, 96, 96],  [1, 1, 1, 1])



        concat = tf.concat([conv_1_1, conv_2_1, conv_3_2, conv_4_3], axis=3, name='concat')


        #self.print_activations(concat)

        return concat



    def reduction_A(self,tensor):

        '''

        Generates the graph for the Reduction A subgraph of the Inception v4 model

        '''

        pool_1_1 = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name = 'pool_1_1')



        conv_2_1 = Model.conv(tensor,  'conv_2_1', [3, 3, 384, 384], [1, 2, 2, 1], padding='VALID')



        conv_3_1 = Model.conv(tensor,  'conv_3_1', [1, 1, 384, 192], [1, 1, 1, 1])

        conv_3_2 = Model.conv(conv_3_1, 'conv_3_2', [3, 3, 192, 224], [1, 1, 1, 1])

        conv_3_3 = Model.conv(conv_3_2, 'conv_3_3', [3, 3, 224, 256], [1, 2, 2, 1], padding='VALID')



        concat = tf.concat([pool_1_1, conv_2_1, conv_3_3], axis=3, name = 'concat')


        #self.print_activations(concat)


        return concat



    def inception_B(self,tensor):

        '''

        Generates the graph for the Inception-B subgraph of the Inception v4 model

        '''

        pool_1_1 = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name = 'pool_1_1')

        conv_1_2 = Model.conv(pool_1_1, 'conv_1_2', [1, 1, 1024, 128], [1, 1, 1, 1],)



        conv_2_1 = Model.conv(tensor,  'conv_2_1', [1, 1, 1024, 384], [1, 1, 1, 1])



        conv_3_1 = Model.conv(tensor,  'conv_3_1', [1, 1, 1024, 192], [1, 1, 1, 1])

        conv_3_2 = Model.conv(conv_3_1, 'conv_3_2', [7, 1, 192, 224],  [1, 1, 1, 1])

        conv_3_3 = Model.conv(conv_3_2, 'conv_3_3', [1, 7, 224, 256],  [1, 1, 1, 1])



        conv_4_1 = Model.conv(tensor,  'conv_4_1', [1, 1, 1024, 192], [1, 1, 1, 1])

        conv_4_2 = Model.conv(conv_4_1, 'conv_4_2', [1, 7, 192, 192],  [1, 1, 1, 1])

        conv_4_3 = Model.conv(conv_4_2, 'conv_4_3', [7, 1, 192, 224],  [1, 1, 1, 1])

        conv_4_4 = Model.conv(conv_4_3, 'conv_4_4', [1, 7, 224, 224],  [1, 1, 1, 1])

        conv_4_5 = Model.conv(conv_4_4, 'conv_4_5', [7, 1, 224, 256],  [1, 1, 1, 1])



        concat = tf.concat([conv_1_2, conv_2_1, conv_3_3, conv_4_5], axis=3, name = 'concat')


        #self.print_activations(concat)


        return concat



    def reduction_B(self,tensor):

        '''

        Generates the graph for the Reduction-B subgraph of the Inception v4 model

        '''

        pool_1_1 = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name = 'pool_1_1')



        conv_2_1 = Model.conv(tensor,  'conv_2_1', [1, 1, 1024, 192], [1, 1, 1, 1])

        conv_2_2 = Model.conv(conv_2_1, 'conv_2_2', [3, 3, 192, 192],  [1, 2, 2, 1], padding='VALID')



        conv_3_1 = Model.conv(tensor,  'conv_3_1', [1, 1, 1024, 256], [1, 1, 1, 1])

        conv_3_2 = Model.conv(conv_3_1, 'conv_3_2', [1, 7, 256, 256],  [1, 1, 1, 1])

        conv_3_3 = Model.conv(conv_3_2, 'conv_3_3', [7, 1, 256, 320],  [1, 1, 1, 1])

        conv_3_4 = Model.conv(conv_3_3, 'conv_3_4', [3, 3, 320, 320],  [1, 2, 2, 1], padding='VALID')



        concat = tf.concat([pool_1_1, conv_2_2, conv_3_4], axis=3, name = 'concat')


        #self.print_activations(concat)


        return concat



    def inception_C(self,tensor):

        '''

        Generates the graph for the Inception-C subgraph of the Inception v4 model

        '''

        pool_1_1   = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name = 'pool_1_1')

        conv_1_2   = Model.conv(pool_1_1, 'conv_1_2',   [1, 1, 1536, 256], [1, 1, 1, 1])



        conv_2_1   = Model.conv(tensor,  'conv_2_1',   [1, 1, 1536, 256], [1, 1, 1, 1])



        conv_3_1   = Model.conv(tensor,  'conv_3_1',   [1, 1, 1536, 384], [1, 1, 1, 1])

        conv_3_2_1 = Model.conv(conv_3_1, 'conv_3_2_1', [1, 3, 384, 256],  [1, 1, 1, 1])

        conv_3_2_2 = Model.conv(conv_3_1, 'conv_3_2_2', [3, 1, 384, 256],  [1, 1, 1, 1])



        conv_4_1   = Model.conv(tensor,  'conv_4_1',   [1, 1, 1536, 384], [1, 1, 1, 1])

        conv_4_2   = Model.conv(conv_4_1, 'conv_4_2',   [1, 3, 384, 448],  [1, 1, 1, 1])

        conv_4_3   = Model.conv(conv_4_2, 'conv_4_3',   [3, 1, 448, 512],  [1, 1, 1, 1])

        conv_4_3_1 = Model.conv(conv_4_3, 'conv_4_3_1', [1, 3, 512, 256],  [1, 1, 1, 1])

        conv_4_3_2 = Model.conv(conv_4_3, 'conv_4_3_2', [3, 1, 512, 256],  [1, 1, 1, 1])



        concat = tf.concat([conv_1_2, conv_2_1, conv_3_2_1, conv_3_2_2, conv_4_3_1, conv_4_3_2], axis=3, name = 'concat')


        #self.print_activations(concat)

        return concat








    def conv(tensor, name, shape, strides=[1, 1, 1, 1], padding='SAME', activation=tf.nn.relu):

        """

        Generates a convolutional layer



        Keyword arguments:

        tensor -- input tensor. Must be indexed by [sample, row, col, ch]

        name -- the name that will be given to the tensorflow Variable in the GraphDef

        shape -- the shape of the kernel. Must be indexed by [row, col, num_input_ch, num_output_ch]

        strides -- the stride of the convolution. Must be indexed by [sample, row, col, ch]

        padding -- if set to 'SAME', the output will have the same height and width as the input. If

                set to 'VALID', the output will have its size reduced by the difference between the

                tensor size and kernel size

        activation -- the activation function to use


        kernel = tf.Variable(tf.truncated_normal([11,11,3,96],dtype = tf.float32,
                                                 stddev = 1e-1),name = 'weights')
        conv = tf.nn.conv2d(inputs,kernel,[1,4,4,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,shape = [96],dtype = tf.float32),
                                         trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name = scope)
        """





        W = tf.get_variable(name+"_W", shape,initializer = tf.contrib.layers.variance_scaling_initializer())
        #W = tf.Variable(tf.truncated_normal(shape,dtype = tf.float32,stddev = 1e-1),name = name+'_W')

        b = tf.get_variable(name+"_b", shape[-1],initializer=tf.constant_initializer(0.01))
        #b = tf.Variable(tf.constant(0.0,shape = shape[-1],dtype = tf.float32),trainable = True, name = name+'_b')


        tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', W)

        tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', b)

        z = tf.nn.conv2d(tensor, W, strides=strides, padding=padding, name=name+'_conv')

        h = tf.nn.bias_add(z, b)

        a = activation(h, name=name+'_conv_bias_activate')



        return a







    def dense(self, tensor, name, num_out):

        '''

        Generates a fully connected layer. Does not apply an activation function



        Keyword arguments:

        tensor -- input tensor. Must be indexed by [sample, ch]

        name -- the name that will be given to the tensorflow Variable in the GraphDef

        num_out -- the size of the output tensor

        '''
        #shape = int(np.prod(tensor.get_shape()[1:]))

        W_fc = tf.get_variable('W_fc', [tensor.shape[1], num_out],initializer = tf.contrib.layers.variance_scaling_initializer())
        #W_fc = tf.Variable(tf.truncated_normal([shape,num_out],dtype = tf.float32,stddev = 1e-1),name = name+'_W')


        b_fc = tf.get_variable('b_fc', [num_out],initializer=tf.constant_initializer(0.01))
        #b_fc = tf.Variable(tf.constant(0.0,shape = [num_out],dtype = tf.float32),trainable = True,name = name+'_b')


        tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', W_fc)

        tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', b_fc)



        z_fc = tf.matmul(tensor, W_fc, name='fc_mat')

        h_fc = tf.nn.bias_add(z_fc, b_fc)

        #self.print_activations(h_fc)

        return h_fc



    def train(self,batch_size = 128,epochs = 100000,keep_prob = 0.8):
        '''

        Defines the variables necessary for training then begins training


        Keyword arguments:

        generator -- a data_generator object with an implementation of get_batch, as seen

                          in the data_generator.py module

        batch_size -- the number of samples to be used in each training batch. Keep memory

                      constraints in mind

        epochs -- the number of epochs to be used for training

        keep_prob -- the keep probability of the dropout layer to be used for training

        '''
        #self.cross_entropy = tf.reduce_sum(tf.square(self.y_hat-self.y))
        self.lr = 1e-5
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.y, logits = self.fc))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        #params = tf.trainable_variables()
        #opt = tf.train.AdamOptimizer(learning_rate=0.001)
        #gradients = tf.gradients(self.cross_entropy, params)
        #clipped_gradients, norm = tf.clip_by_global_norm(gradients,5)
        #train_op = opt.apply_gradients(zip(clipped_gradients, params))

        self.correct_prediction = tf.equal(tf.argmax(self.y_hat,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        # Creates a saver object to generate checkpoints during training. This one also saves

        # the gradients and the increment of the Adam Optimizer.

        ckptPath = self.ckptPath
        modelPath = self.modelPath
        self.saver = tf.train.Saver()


        l,acc,valid_acc = [],[],[]
        g = generator('./trainset.csv',batch_size)
        v = generator('./validset.csv',batch_size)
        with tf.Session() as sess:
            if os.path.isdir('trainModel'):
                self.saver.restore(sess,ckptPath)

            else:
                sess.run(tf.global_variables_initializer())

            for i in range(epochs):
                bag = next(g)
                images,labels = bag[0],bag[1]


                label = (np.arange(self.numclass) == labels[:,None]).astype('float32')

                #logit = self.y_hat.eval(feed_dict = {self.x:images,self.keep_prob:1.0})


                loss = self.cross_entropy.eval(feed_dict={self.x: images, self.y: label, self.keep_prob: 1.0})
                train_accuracy = self.accuracy.eval(feed_dict={self.x: images, self.y: label, self.keep_prob: 1.0})




                #print('step %d, loss %3f, training accuracy %.3f' % (i, loss, train_accuracy))

                if i % 50 == 0:
                    #获取权重和偏置
                    #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
                    #tf.get_default_graph().get_tensor_by_name(variable_name)
                    print('step %d, loss %3f, training accuracy %.3f' % (i, loss, train_accuracy))
                    l.append(loss)
                    acc.append(train_accuracy)


                if i % 20000 == 0 and i != 0:
                    self.lr *= 0.1




                if i % 1407 == 0:
                    valid_bag = next(v)
                    valid_imgs,valid_labels = valid_bag[0],valid_bag[1]
                    valid_label = (np.arange(self.numclass) == valid_labels[:,None]).astype('float32')

                    valid_accuracy = self.accuracy.eval(feed_dict={self.x: valid_imgs, self.y: valid_label, self.keep_prob: 1.0})
                    valid_acc.append(valid_accuracy)
                    print('step %d, validing accuracy %.3f' % (i, valid_accuracy))
                    self.saver.save(sess, save_path = ckptPath)

                self.train_step.run(feed_dict={self.x: images, self.y: label, self.keep_prob: 1.0})

            self.model_saver.save(sess, save_path = modelPath)
            np.save('loss.npy',l)
            np.save('train_acc.npy',acc)
            np.save('valid_acc.npy',valid_acc)







    def classify(self, image):

        '''

        Classifies the input image based on the trained model in modelPath


        Keyword arguments:

        image -- the image, or image array, indexed by [sample, row, col, ch]

        '''

        with tf.Session() as sess:
            modelPath = self.modelPath
            self.model_saver.restore(sess, save_path = modelPath)


            y = self.y_hat.eval(feed_dict={self.x: image, self.keep_prob: 1.0})




        return y






    def verification(self,image):
        with tf.Session() as sess:
            modelPath = self.modelPath
            self.model_saver.restore(sess, save_path = modelPath)
            y = self.pool_reshape.eval(feed_dict={self.x: image})

        return y




def read_csv_file(csv_file):
    x, y = [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            path, label = line.strip().split()
            x.append(path)
            y.append(int(label))

    #np.asarray()强制转换成向量并且为指定的数据类型
    return x, np.asarray(y, dtype='int32')


def vectorize_imgs(img_path):  #图片变成向量，return arr_img
    with Image.open(img_path) as img:
        arr_img = np.asarray(img, dtype='float32')
        return arr_img/255.




def main():
    #划分数据集
    #src = 'verification\\false'
    #dataset_split(src)


    #训练网络
    model = Model([None, 299, 299, 3],[None, 151])
    #model.train(batch_size = 64,epochs = 42210)



    #loss = np.load('loss.npy')
    #train_acc = np.load('train_acc.npy')
    #valid_acc = np.load('valid_acc.npy')

    #画损失值
    #plt.figure(figsize=(9, 6), dpi=300)
    #plt.plot(loss)
    #plt.xlabel('train_step')
    #plt.ylabel('loss')
    #plt.title("loss")
    #plt.savefig('loss.png', format='png')

    #画准确率
    #plt.figure(figsize=(9, 6), dpi=300)
    #plt.plot(train_acc,label = 'train_acc')
    #plt.plot(valid_acc,'r',label = 'valid_acc')
    #plt.xlabel('train_step')
    #plt.ylabel('accuracy')
    #plt.title("loss")
    #plt.legend(loc=0)
    #plt.savefig('accuracy.png', format='png')


    #split = dataset_split(src)





    #Verification
    #imgs1, imgs2, veri_labels = split.read_csv_pair_file('./verificate_set.csv')
    imgs,class_label = read_csv_file('validset.csv')


    #same,diff,veri_pred,class_pred = [],[],[],[]

    predicts = []
    for i in range(len(imgs)):
        img = []
        img.append(vectorize_imgs(imgs[i]))
        #result1 = model.verification(np.asarray(vectorize_imgs(imgs1[i]),dtype='float32'))
        #result2 = model.verification(np.asarray(vectorize_imgs(imgs2[i]),dtype='float32'))


        #class1 = np.argmax(model.classify(np.asarray(vectorize_imgs(imgs1[i]),dtype='float32')),axis=1)
        #class2 = np.argmax(model.classify(np.asarray(vectorize_imgs(imgs2[i]),dtype='float32')),axis=1)

        result = np.argmax(model.classify(np.asarray(img,dtype='float32')),axis=1)
        predicts.append(result)
        #veri_pred.append(int(class1 == class2))
        #class_pred.append(result)





        #l2 = np.sqrt(np.sum(np.square(result1-result2)))



        #if i % 100 == 0:
        #    print('L2 %.3f label %d'%(l2,veri_labels[i]))


        #if int(veri_labels[i]) == 1:
        #    same.append(l2)
        #else:
        #    diff.append(l2)


    np.save('predicts.npy',predicts)
    np.save('labels.npy',class_label)
    #veri_pred = np.asarray(veri_pred)

    """
    #测试集准确率
    class_pred = np.asarray(class_pred)
    test_label = (np.arange(151) == class_label[:,None]).astype('float32')
    correct_prediction = np.equal(class_pred, np.argmax(test_label, 1))

    test_accuracy = np.sum(correct_prediction) / correct_prediction.size


    #保存
    np.save('test_accuracy.npy',test_accuracy)
    np.save('same.npy',same)
    np.save('different.npy',diff)


    np.save('veri_pred.npy',veri_pred)
    np.save('verificate_labels.npy',veri_labels)

    np.save('class_pred.npy',class_pred)
    np.save('class_label.npy',class_label)




    #相同图像l2距离
    print('same distance : %.3f' % (float(same/len(same))))
    #不同图像l2距离
    print('different distance : %.3f' % (float(diff/len(diff))))

    #画散点图
    plt.figure(1,figsize=(9, 6), dpi=300)
    plt.scatter(same,same,c = 'b',label = 'same')
    plt.scatter(diff,diff,c = 'r',marker = "x",label = 'different')
    plt.grid(True)
    plt.title("L2 distance")
    plt.legend(loc=0)
    plt.savefig('verification_L2_distance.png', format='png')
    #plt.show()


    #画二分类混淆矩阵
    binary_confusion_mat = confusion_matrix(veri_labels, veri_pred)
    np.save('binary_confusion_mat.npy',binary_confusion_mat)

    tn, fp, fn, tp = binary_confusion_mat.ravel()
    #精确率
    precision = tp/(tp+fp)
    #召回率
    recall = tp/(tp+fn)
    #准确率
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    #特异率
    specificity = tn/(fp+tn)





    #实验分析数据与指标
    analysis = []


    analysis += zip(['Test_classify_accuracy'],[test_accuracy])
    analysis += zip(['Same distance'],[float(same/len(same))])
    analysis += zip(['Different distance'],[float(same/len(same))])
    analysis += zip(['Precision'],[precision])
    analysis += zip(['Recall'],[recall])
    analysis += zip(['Accuracy'],[accuracy])
    analysis += zip(['Specifcity'],[specificity])

    dataset_split.set_to_csv_file(analysis,'analysis.csv')








    #画多分类混淆矩阵
    cm = confusion_matrix(class_label, class_pred)
    np.save('confusion_mat.npy',cm)



    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Paired):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        plt.tick_params(labelsize=10)
        xlocations = np.array(range(len(set(class_label))))
        plt.xticks(xlocations, class_label, rotation=90)
        plt.yticks(xlocations, class_label)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(15, 10), dpi=300)
    ind_array = np.arange(len(set(class_label)))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.0001:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=10, va='center', ha='center')

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig('confusion_matrix.png', format='png')

    """


    """
    predicts,label = [],[]
    for i in range(20):
        test = next(t)
        imgs,labels = test[0],test[1]
        label += labels.tolist()
        result = model.classify(imgs)
        predicts += np.argmax(result, axis=1).tolist()


    np.save('predicts.npy',predicts)
    np.save('label.npy',label)



    predicts = np.load('predicts.npy')
    label = np.load('label.npy')

    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Paired):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        plt.tick_params(labelsize=8)
        xlocations = np.array(range(len(set(label))))
        plt.xticks(xlocations, label, rotation=90)
        plt.yticks(xlocations, label)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')



    cm = confusion_matrix(label, predicts)
    #tn, fp, fn, tp = confusion_matrix(label, predicts).ravel()


    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(24, 16), dpi=300)
    ind_array = np.arange(len(set(label)))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.0001 and c != 1.0:
            plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=6, va='center', ha='center')

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig('confusion_matrix.png', format='png')
    #plt.show()
    """








if __name__ == '__main__':
    main()
