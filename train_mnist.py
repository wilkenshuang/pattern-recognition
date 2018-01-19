# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:47:30 2017

@author: gd
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import modelMine
import cv2
import numpy as np
from skimage import morphology,draw

MNIST_data_folder='C:/Anaconda/image/MNIST_data'
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)

def mnist_process(imgset):
    output=np.zeros((len(imgset),48*48))
    for i in range(len(imgset)):
        test=255*imgset[i].reshape((28,28))
        test=cv2.resize(test,(48,48),cv2.INTER_NEAREST).astype(np.uint8)
        ret,th=cv2.threshold(test,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #skeleton=255*np.asarray(morphology.skeletonize(th/255),dtype='uint8')
        #ret,skele_extrac=cv2.threshold(skeleton,128,255,cv2.THRESH_BINARY_INV)
        output[i]=th.reshape(48*48)
    return output

IMAGE_WIDTH=48
IMAGE_HEIGHT=48
nclass=10
lr_rate=0.0005
#预存模型地址读取
#model_path=os.getcwd()
model_path='C:/Anaconda/image/utils_dir/testModel0'

X=tf.placeholder(tf.float32,[None,IMAGE_WIDTH*IMAGE_HEIGHT])
Y=tf.placeholder(tf.float32,[None,nclass])
keep_prob=tf.placeholder(tf.float32)
lr_base=tf.placeholder(tf.float32)
train_phase=tf.placeholder(tf.bool)
keep_prob=tf.placeholder(tf.float32)

train_logit,l2_loss=modelMine.inference(X,nclass,keep_prob)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=train_logit))
optimizer=tf.train.AdamOptimizer(learning_rate=lr_base).minimize(loss)
correction=tf.equal(tf.argmax(Y,1),tf.argmax(train_logit,1))
accuracy=tf.reduce_mean(tf.cast(correction,tf.float32))
saver=tf.train.Saver()
#ckpt = tf.train.get_checkpoint_state(model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
count=1
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,ckpt.model_checkpoint_path)
    for step in range(3001):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 使用SGD，每次选取200个数据训练
        batch_xs=mnist_process(batch_xs)
        _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_xs, Y: batch_ys,
                            train_phase:True,keep_prob: 0.8,lr_base:lr_rate})
        #print("轮次：%d, Loss: %f" %(step,loss_))
        if step%500==0 and step!=0:#每500次输出一下测试集上准确度
            batch_xs_ts, batch_ys_ts =mnist.test.next_batch(128)
            batch_xs_ts=mnist_process(batch_xs_ts)
            valid_accuracy = accuracy.eval(feed_dict={X:batch_xs_ts,Y:batch_ys_ts,keep_prob:1,
                                                      train_phase:False})
            print("第%d轮,模型在测试集上精准率为：%0.2f%%" %(count,valid_accuracy*100))
            #saver.save(sess,"/Users/Huang/image/utils_dir/model.ckpt",global_step=step)
            count+=1
    saver.save(sess,"C:/Anaconda/image/utils_dir/testModel/model.ckpt",global_step=step)
