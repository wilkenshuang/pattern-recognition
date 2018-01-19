# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:47:30 2017

@author: gd
"""


import modelMine
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
#import matplotlib.pyplot as plt
import cv2
import PIL.Image as Image
import pandas as pd
from skimage import morphology,draw
from collections import Counter

trimg_dir='C:/Anaconda/image/train'
teimg_dir='C:/Anaconda/image/test'
Type={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,
       '9':9,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,
       'S':18,'T':19}

types=['0','1','2','3','4','5','6','7','8',
       '9']

f_tr = open('C:/Anaconda/image/train_pic.csv', encoding='utf-8')
f_te = open('C:/Anaconda/image/test_pic.csv', encoding='utf-8')

TRAIN=pd.read_csv(f_tr)
TEST=pd.read_csv(f_te)

def preprocess(dataset):
    Label=[]
    IMG_set=np.zeros((len(dataset),48,48))
    #从csv文件里读取参数
    for i in dataset.values:
        img_path=i[1]

        categ=Type[str(i[2])]
        count=i[0]
        image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    #建立图片集和标签集
        img=cv2.resize(image,(48,48),interpolation=cv2.INTER_NEAREST)
        ret,th=cv2.threshold(img,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret,th=cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #skeleton=255*np.asarray(morphology.skeletonize(th/255),dtype='uint8')
        #ret,skele_extrac=cv2.threshold(skeleton,128,255,cv2.THRESH_BINARY_INV)
        IMG_set[count]=th
        Label.append(categ)

    return IMG_set,Label
sess = tf.InteractiveSession()

Tr_img,Tr_lab=preprocess(TRAIN)#将2D图像维度降为1D
Te_img,Te_lab=preprocess(TEST)

Tr_img=np.reshape(Tr_img,(-1,48*48))#-1代表自适应未指定值
Te_img=np.reshape(Te_img,(-1,48*48))

Tr_lab=np.array(Tr_lab,dtype=np.uint8)
Te_lab=np.array(Te_lab,dtype=np.uint8)
#定义参数
IMAGE_WIDTH,IMAGE_HEIGHT=48,48
split_param=0.1
nclasses=10
reg_alpha=0.0001
#对标签进行多维映射
Tr_lab= (np.arange(nclasses) == Tr_lab[:,None]).astype(np.float32)
Te_lab= (np.arange(nclasses) == Te_lab[:,None]).astype(np.float32)
#划分训练集和验证集
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
#    IMAGE_WIDTH=48
#    IMAGE_HEIGHT=48
    lr_rate=0.0005
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
    ckpt = tf.train.get_checkpoint_state(model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True#使用allow_growth option，刚一开始分配少量的GPU容量，
                                          #然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #sess=tf.Session(config=config)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if ckpt:
            saver.restore(sess,ckpt.model_checkpoint_path)
        count=1
        #while True:
        for step in range(1501):
            batch_x_tr, batch_y_tr = get_batch(x_train,y_train,batch_size=50)
            #optimizer.run(feed_dict={X:batch_x_tr,Y:batch_y_tr, keep_prob:0.8,train_phase:True})
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x_tr, Y: batch_y_tr,
                               train_phase:True,keep_prob: 0.75,lr_base:lr_rate})

            #print("轮次：%d, Loss: %f" %(step,loss_))

            if step%100==0 and step!=0:
                batch_x_te, batch_y_te = get_batch(Te_img,Te_lab,batch_size=50)
                valid_accuracy = accuracy.eval(feed_dict={X:batch_x_te,Y:batch_y_te,keep_prob:1,
                                                          train_phase:False})
                #valid_accuracy = sess.run(accuracy,feed_dict={X:batch_x_va,Y:batch_y_va,keep_prob:1,
                #                                          train_phase:False})
                print("第%d轮,模型在测试集上精准率为：%d%%" %(count,valid_accuracy*100))
                if valid_accuracy>0.97 and step>500:
                    break
                count+=1
        saver.save(sess,'C:/Anaconda/image/utils_dir/testModel/model.ckpt',global_step=step)
