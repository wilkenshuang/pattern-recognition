# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:41:56 2017

@author: gd
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
#import matplotlib.pyplot as plt
from skimage import morphology,draw
from sklearn.model_selection import train_test_split
import modelMine

lianbi='C:/Anaconda/image/multi-train'
#lianbi='/Users/Huang/image/multi-train'
types=['0','1','2','3','4','5','6','7','8',
       '9']

IMG_set=np.zeros((918,48,48))
count=0
Label=[]
for i in types:
    for j in os.listdir(os.path.join(lianbi,i)):
        ID=j

        img_path=os.path.join(lianbi,i,ID)
        image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        ret,th=cv2.threshold(image,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        h,w=image.shape
        if h > 1.5*w:
            #ret,th=cv2.threshold(image,128,255,cv2.THRESH_BINARY_INV)
            offset=(h//1.5-w)//2
            mask=255*np.ones((h,int(offset))).astype(np.uint8)
            th=np.concatenate((mask,th,mask),axis=1)
            #img=cv2.resize(image,(48,48),interpolation=cv2.INTER_AREA)
        img=cv2.resize(th,(48,48),interpolation=cv2.INTER_NEAREST)
        ret,th=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        #skeleton=255*np.asarray(morphology.skeletonize(th/255),dtype='uint8')
        #ret,skele_extrac=cv2.threshold(skeleton,128,255,cv2.THRESH_BINARY_INV)
        IMG_set[count]=th
        Label.append(int(i))
        count+=1

nclasses=10
split_param=0.1
IMAGE_WIDTH,IMAGE_HEIGHT=48,48

Tr_img=np.reshape(IMG_set,(-1,48*48))
Tr_lab=np.array(Label,dtype=np.uint8)
Tr_lab= (np.arange(nclasses) == Tr_lab[:,None]).astype(np.float32)

x_train,x_valid,y_train,y_valid=train_test_split(Tr_img,Tr_lab,test_size=split_param)

def get_batch(imgset,labset,batch_size=200):
    batch_x=np.zeros([batch_size,IMAGE_WIDTH*IMAGE_HEIGHT])
    batch_y=np.zeros([batch_size,10])
    for i in range(batch_size):
        num=np.random.randint(1,imgset.shape[0])
        batch_x[i]=imgset[num]
        batch_y[i]=labset[num]
    return batch_x,batch_y

with tf.Graph().as_default():
    IMAGE_WIDTH=48
    IMAGE_HEIGHT=48
    nclass=10
    lr_rate=0.0001
    #预存模型地址读取
    model_path='C:/Anaconda/image/utils_dir/testModel'
    X=tf.placeholder(tf.float32,[None,IMAGE_WIDTH*IMAGE_HEIGHT])
    Y=tf.placeholder(tf.float32,[None,nclasses])
    keep_prob=tf.placeholder(tf.float32)
    lr_base=tf.placeholder(tf.float32)
    train_phase=tf.placeholder(tf.bool)
    train_logit,l2_loss=modelMine.inference(X,nclasses,keep_prob)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=train_logit))#+\
                        #reg_alpha*l2_loss
    optimizer=tf.train.AdamOptimizer(learning_rate=lr_base).minimize(loss)
    correction=tf.equal(tf.argmax(Y,1),tf.argmax(train_logit,1))
    accuracy=tf.reduce_mean(tf.cast(correction,tf.float32))
    saver=tf.train.Saver()
    #ckpt = tf.train.get_checkpoint_state(model_path)
    ckpt=tf.train.latest_checkpoint(model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True#使用allow_growth option，刚一开始分配少量的GPU容量，
                                          #然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #sess=tf.Session(config=config)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if ckpt:
            saver.restore(sess,ckpt)
        count=1
        #while True:
        for step in range(1501):
            batch_x_tr, batch_y_tr = get_batch(x_train,y_train,batch_size=50)
            #optimizer.run(feed_dict={X:batch_x_tr,Y:batch_y_tr, keep_prob:0.8,train_phase:True})
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x_tr, Y: batch_y_tr,
                               train_phase:True,keep_prob: 0.75,lr_base:lr_rate})
            #print("轮次：%d, Loss: %f" %(step,loss_))
            if step%100==0:
                batch_x_va, batch_y_va = get_batch(x_valid,y_valid,batch_size=200)
                valid_accuracy = accuracy.eval(feed_dict={X:batch_x_va,Y:batch_y_va,keep_prob:1,
                                                                      train_phase:False})
                print("第%d轮,模型在测试集上精准率为：%d%%" %(step,valid_accuracy*100))
                if valid_accuracy>0.98 and step>100:
                    break
        saver.save(sess,'C:/Anaconda/image/utils_dir/testModel/model.ckpt',global_step=step)

        '''
        sum=0
        for i in range(10):
            batch_x_va, batch_y_va = get_batch(x_valid,y_valid,batch_size=200)
            valid_accuracy = accuracy.eval(feed_dict={X:batch_x_va,Y:batch_y_va,keep_prob:1,
                                                      train_phase:False})
            print("第%d轮,模型在验证集上精准率为：%0.2f%%" %(i+1,valid_accuracy*100))
            sum+=valid_accuracy*100
        print("平均识别精确率: %0.2f%%" %(sum/10))
        '''
