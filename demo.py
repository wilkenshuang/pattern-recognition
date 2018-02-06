# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:42:07 2017

@author: gd
"""
from xmlrpc.server import SimpleXMLRPCServer,SimpleXMLRPCRequestHandler
import socketserver
from socketserver import TCPServer
from socketserver import ThreadingMixIn
import threading
import sys
import tensorflow as tf
import numpy as np
import cv2
import heapq
import os
import profile
import base64
import time
import io
import json
import base64
from PIL import Image
#import matplotlib.pyplot as plt
from skimage import morphology,draw
from collections import Counter
from itertools import groupby
from PIL import ImageEnhance
from PIL import Image

def _cfs_v(image):
    #ret,th=cv2.threshold(image,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th=cv2.bilateralFilter(th,3,50,50)
    """传入二值化后的图片进行连通域分割"""
    #img=th.T
    img=image
    h,w= img.shape
    visited = set()
    q = queue.Queue()
    offset = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    cuts = []
    for y in range(w):
        for x in range(h):
            y_axis = []
            #y_axis = []
            if img[x,y] == 0 and (x,y) not in visited:
                q.put((x,y))
                visited.add((x,y))
            while not q.empty():
                x_p,y_p = q.get()
                for x_offset,y_offset in offset:
                    x_c,y_c = x_p+x_offset,y_p+y_offset
                    if (x_c,y_c) in visited:
                        continue
                    visited.add((x_c,y_c))
                    try:
                        if img[x_c,y_c] == 0:
                            q.put((x_c,y_c))
                            y_axis.append(y_c)
                            #y_axis.append(y_c)
                    except:
                        pass
            if y_axis:
                min_y,max_y = min(y_axis),max(y_axis)
                if max_y - min_y >  100:
                    # 宽度小于3的认为是噪点，根据需要修改
                    cuts.append((min_y,max_y))
    return cuts

def _cfs_h(img):
    """传入二值化后的图片进行连通域分割"""
    img=img.T
    w,h = img.shape
    visited = set()
    q = queue.Queue()
    offset = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    cuts = []
    for x in range(w):
        for y in range(h):
            #x_axis = []
            y_axis = []
            if img[x,y] == 0 and (x,y) not in visited:
                q.put((x,y))
                visited.add((x,y))
            while not q.empty():
                x_p,y_p = q.get()
                for x_offset,y_offset in offset:
                    x_c,y_c = x_p+x_offset,y_p+y_offset
                    if (x_c,y_c) in visited:
                        continue
                    visited.add((x_c,y_c))
                    try:
                        if img[x_c,y_c] == 0:
                            q.put((x_c,y_c))
                            #x_axis.append(x_c)
                            y_axis.append(y_c)
                    except:
                        pass
            if y_axis:
                min_y,max_y = min(y_axis),max(y_axis)
                if max_y - min_y >=  0.1*w:
                    # 宽度小于3的认为是噪点，根据需要修改
                    cuts.append((min_y,max_y))
    return cuts

#去噪
def _denoise(img,times=1):
    w,h=img.shape
    for i in range(times):
        for i in range(1,w-1):
            for j in range(1,h-1):
                count=0
                if img[i-1][j-1]>245:
                    count+=1
                if img[i][j-1]>245:
                    count+=1
                if img[i+1][j-1]>245:
                    count+=1
                if img[i-1][j]>245:
                    count+=1
                if img[i+1][j]>245:
                    count+=1
                if img[i-1][j+1]>245:
                    count+=1
                if img[i][j+1]>245:
                    count+=1
                if img[i][j+1]>245:
                    count+=1
                if count>4:
                    img[i][j]=255
    return img

#垂直投影
def _vertical(img):
    h,w=img.shape
    ret,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    """传入二值化后的图片进行连通域分割"""
    ver_list=[]
    for i in range(w):
        black=0
        for j in range(h):
            if img[j][i]==0:
                black+=1
        ver_list.append(black)
    #判断边界
    l,r=0,0
    flag=False
    cut_v=[]
    for i,count in enumerate(ver_list):
        if flag is False and count>5:
            l=i
            flag=True
        if flag and count<=2:
            r=i-1
            flag=False
        #if r-l>=5:
            cut_v.append((l,r))
    return cut_v

def _horizontal(img):
    h,w=img.shape
    ret,img=cv2.threshold(img,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    """传入二值化后的图片进行连通域分割"""
    ver_list=[]
    for i in range(h):
        black=0
        for j in range(w):
            if img[i][j]==0:
                black+=1
        ver_list.append(black)
    #判断边界
    t,b=0,0
    flag=False
    cut_h=[]
    for i,count in enumerate(ver_list):
        if flag is False and count>4:
            t=i
            flag=True
        if flag and count<=2:
            b=i-1
            flag=False
        #if b-t>=5:
            cut_h.append((t,b))
    return cut_h

def _projection(img):
    h,w=img.shape
    result=[]
    for y in range(w):
        black=0
        for x in range(h):
            if img[x,y]==0:
            #if img[x,y]==255:
                black+=1
        result.append(black)
    return result

def get_start(hist_width):

    mid=len(hist_width)//2
    temp=hist_width[mid-10:mid+10]
    return mid-int(0.1*len(hist_width))+np.argmin(temp)

def get_nearby_pixel(img,x,y,num):
    if num==1:
        return 0 if img[x+1,y-1]==0 else 1
    if num==2:
        return 0 if img[x+1,y]==0 else 1
    if num==3:
        return 0 if img[x+1,y+1]==0 else 1
    if num==4:
        return 0 if img[x,y+1]==0 else 1
    if num==5:
        return 0 if img[x,y-1]==0 else 1
    else:
        raise Exception("get_nearby_pix_value error")

def get_end_route(img,start_x,height):
    left_limit=0
    right_limit=img.shape[1]-1
    end_route=[]
    cur_p=(0,start_x)
    last_p=cur_p
    end_route.append(cur_p)

    while cur_p[0]<(height-1):
        sum_n=0
        max_w=0
        next_x=cur_p[0]
        next_y=cur_p[1]
        for i in range(1,6):
            cur_w=get_nearby_pixel(img,cur_p[0],cur_p[1],i)*(6-i)
            sum_n+=cur_w
            if max_w<cur_w:
                max_w=cur_w

        # 如果全黑，需要看惯性
        if sum_n==0:
            max_w=4

        # 如果全白，则默认垂直下落
        if sum_n==15:
            max_w=6

        if max_w==1:
            next_x=cur_p[0]
            next_y=cur_p[1]-1
        elif max_w==2:
            next_x=cur_p[0]
            next_y=cur_p[1]+1
        elif max_w==3:
            next_x=cur_p[0]+1
            next_y=cur_p[1]+1
        elif max_w==5:
            next_x=cur_p[0]+1
            next_y=cur_p[0]-1
        elif max_w==6:
            next_x=cur_p[0]+1
            next_y=cur_p[1]
        elif max_w==4:
            if next_y>cur_p[1]:# 具有向右的惯性
                next_x=cur_p[0]+1
                next_y=cur_p[1]+1
            if next_y<=cur_p[1]:# 垂直下落
                next_x=cur_p[0]+1
                next_y=cur_p[1]
            if sum_n==0:# 垂直下落
                next_x=cur_p[0]+1
                next_y=cur_p[1]
        #else:
        #    raise Exception("get end route error")

        # 如果出现重复运动
        if last_p[0]==next_x and last_p[1]==next_y:
            if next_y<cur_p[1]:
                max_w=5
                next_x=cur_p[0]+1
                next_y=cur_p[1]+1
            else:
                max_w=3
                next_x=cur_p[0]+1
                next_y=cur_p[1]-1
        last_p=cur_p

        if next_y>right_limit:
            next_y=right_limit
            next_x=cur_p[0]+1
        if next_y<left_limit:
            next_y=left_limit
            next_x=cur_p[0]+1
        cur_p=(next_x,next_y)
        end_route.append(cur_p)

    return end_route

def start_split(image,start_point,end_route):
    left=start_point[0][1]
    top=start_point[0][0]
    right=end_route[0][1]
    bottom=end_route[0][0]
    for i in range(len(start_point)):
        left=min(left,start_point[i][1])
        top=min(top,start_point[i][0])
        right=max(right,end_route[i][1])
        bottom=max(bottom,end_route[i][0])
    w=right-left+1
    h=bottom-top+1
    return h,w

#输入原图进行数字定位
def segment_direct(image):
    H,W=image.shape
    '''输入的图片是经二值化处理的'''
    ret,th_mask=cv2.threshold(image,230,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret,cont1,hier1=cv2.findContours(th_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    count=0
    res=[]
    for i in cont1:
        x, y, w, h = cv2.boundingRect(i)
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if w*h>100:
            res.append([x,y,w,h])
    res=sorted(res,key=lambda value:value[0])
    '''
    output=[]

    for i in range(len(res)):
        x,y,w,h=res[i]
        test_img=img_d[y:y+h,x:x+w]
        output.append(test_img)
        #test_img=cv2.resize(test_img,(IMAGE_WIDTH,IMAGE_HEIGHT),
        #                    interpolation=cv2.INTER_AREA)
    #plt.imshow(test_img,cmap='gray')

    return output
    '''
    return res

#len(hist_width),length

def dropfall(image):
    #image=segment(image_path)
    h,w=image.shape
    #ret,th=cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV)
    #skeleton=255*np.asarray(morphology.skeletonize(th/255),dtype='uint8')
    #ret,skele_extrac=cv2.threshold(skeleton,150,255,cv2.THRESH_BINARY_INV)

    hist_width=_projection(image)
    startX=get_start(hist_width)
    if 0 not in image[:,startX]:
        coords=(h,startX)
    else:
        start_route=[]
        for x in range(h):
            start_route.append((x,0))
        end_route=get_end_route(image,startX,h)
        #filter_end_route=[max(list(k)) for _,k in groupby(end_route,lambda x:x[0])]
        #r_limit,b_limit=0,0
        #while r_limit<w:
        #coords=start_split(image,start_route,filter_end_route)
    #img1=image[:coords[0],:coords[1]]
    #img2=image[:coords[0],coords[1]:]
    img1=image[:,:end_route[-1][1]]
    img2=image[:,end_route[-1][1]:]
    return img1,img2


#构造深度学习模型

def inference(X,nclasses,keep_prob,w_alpha=0.01,b_alpha=0.05,
              reg_alpha=0.01,IMAGE_HEIGHT=48,IMAGE_WIDTH=48):
    #train_phase训练时是True，测试时是False
    x=tf.reshape(X,shape=[-1,IMAGE_WIDTH,IMAGE_HEIGHT,1])

    #4层卷积层
    w_c1=tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    b_c1=tf.Variable(w_alpha*tf.random_normal([32]))
    conv1=tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1)
    #conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]),
    #                   tf.random_normal(shape=[32], mean=1.0, stddev=0.02),
    #                   train_phase, scope='bn_1')
    conv1=tf.nn.relu(conv1)
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1=tf.nn.dropout(conv1,keep_prob=keep_prob)

    w_c2=tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
    b_c2=tf.Variable(w_alpha*tf.random_normal([64]))
    conv2=tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2)
    #conv2 = batch_norm(conv2, tf.constant(0.0, shape=[64]),
    #                   tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
    #                   train_phase, scope='bn_2')
    conv2=tf.nn.relu(conv2)
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2=tf.nn.dropout(conv2,keep_prob=keep_prob)

    w_c3=tf.Variable(w_alpha*tf.random_normal([3,3,64,128]))
    b_c3=tf.Variable(w_alpha*tf.random_normal([128]))
    conv3=tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3)
    #conv3 = batch_norm(conv3, tf.constant(0.0, shape=[128]),
    #                   tf.random_normal(shape=[128], mean=1.0, stddev=0.02),
    #                   train_phase, scope='bn_3')
    conv3=tf.nn.relu(conv3)
    conv3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3=tf.nn.dropout(conv3,keep_prob=keep_prob)

    w_c4=tf.Variable(w_alpha*tf.random_normal([5,5,128,256]))
    b_c4=tf.Variable(w_alpha*tf.random_normal([256]))
    conv4=tf.nn.bias_add(tf.nn.conv2d(conv3,w_c4,strides=[1,1,1,1],padding='SAME'),b_c4)
    #conv4 = batch_norm(conv4, tf.constant(0.0, shape=[256]),
    #                   tf.random_normal(shape=[256], mean=1.0, stddev=0.02),
    #                   train_phase, scope='bn_4')
    conv4=tf.nn.relu(conv4)
    #conv4=tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv4=tf.nn.dropout(conv4,keep_prob=keep_prob)

    #2层全连接层
    w_fully=tf.Variable(w_alpha*tf.random_normal([6*6*256,1024]))
    b_fully=tf.Variable(w_alpha*tf.random_normal([1024]))
    dense=tf.reshape(conv4,[-1,w_fully.get_shape().as_list()[0]])
    dense=tf.nn.relu(tf.matmul(dense,w_fully)+b_fully)
    dense=tf.nn.dropout(dense,keep_prob=keep_prob)

    w_out=tf.Variable(w_alpha*tf.random_normal([1024,nclasses]))
    b_out=tf.Variable(b_alpha*tf.random_normal([nclasses]))
    output=tf.matmul(dense,w_out)+b_out
    l2_loss=tf.nn.l2_loss(w_c1)+tf.nn.l2_loss(w_c2)+tf.nn.l2_loss(w_c3)+\
            tf.nn.l2_loss(w_c4)+tf.nn.l2_loss(w_fully)+tf.nn.l2_loss(w_out)
    return output,l2_loss

def IMGprocess(inputs,IMAGE_WIDTH=48,IMAGE_HEIGHT=48):
    #if type(inputs)==str:
    #    image=cv2.imread(inputs,cv2.IMREAD_GRAYSCALE)
    #else:
    #    image=inputs
    image=inputs
    ret,th=cv2.threshold(image,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h,w=image.shape
    if h > 1.5*w:
        offset=(h//1.5-w)//2
        mask=255*np.ones((h,int(offset))).astype(np.uint8)
        th=np.concatenate((mask,th,mask),axis=1)
    img=cv2.resize(th,(48,48),interpolation=cv2.INTER_AREA)
    ret,th=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    #skeleton=255*np.asarray(morphology.skeletonize(th/255),dtype='uint8')
    #ret,skele_extrac=cv2.threshold(skeleton,128,255,cv2.THRESH_BINARY_INV)

    return th

def predict(inputs,model_path):
    with tf.Graph().as_default():
        IMAGE_WIDTH=48
        IMAGE_HEIGHT=48
        nclass=10
        res=[]
        index=[]
        X=tf.placeholder(tf.float32,shape=[None,IMAGE_WIDTH*IMAGE_HEIGHT])
        keep_prob=tf.placeholder(tf.float32)
        logit,ret = inference(X,nclass,keep_prob)
        logit=tf.nn.softmax(logit)
        #output = tf.nn.top_k(logit, 3)
        saver=tf.train.Saver()
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(len(inputs)):
            x=np.reshape(inputs[i],(1,len(inputs[i])))
            category = sess.run(logit, feed_dict={X: x, keep_prob: 1})
            categories=heapq.nlargest(3,category[0])
            ind=heapq.nlargest(3,range(len(category[0])),category[0].take)
            res.append(categories)
            index.append(ind)
        sess.close()
        return res ,index

def color_recognize(inputs):
    IMAGE_WIDTH,IMAGE_HEIGHT=48,48
    types=['0','1','2','3','4','5','6','7','8',
           '9','A','B','C','D','E','F','G','H',
           'S','T']
    model_path='/Users/Huang/image/utils_dir0'
    output=0
    data=np.zeros([len(inputs),IMAGE_WIDTH*IMAGE_HEIGHT])
    for i in range(len(inputs)):
        letter=IMGprocess(inputs[i])
        letter=np.array(letter)
        letter=np.reshape(letter,(-1,IMAGE_WIDTH*IMAGE_HEIGHT))
        data[i]=letter
    prob,index=predict(data,model_path)
    for i in range(len(index)):
        output+=10**(len(index)-i-1)*index[i][0]
    #output+=10*index[0][0]+index[1][0]
        #dic={}
        #for i in range(len(prob[0])):
            #dic[types[index[0][i]]]=float(prob[0][i]*100)
            #dict=sorted(dic.items(),key=lambda dic:dic[1],reverse=True)
    #res=json.dumps(dic)

    return output

def color_single(inputs):
    IMAGE_WIDTH,IMAGE_HEIGHT=48,48
    types=['0','1','2','3','4','5','6','7','8',
           '9','A','B','C','D','E','F','G','H',
           'S','T']
    model_path='/Users/Huang/image/utils_dir0'
    mark=0
    #if type(inputs)==list:
        #data=np.zeros([len(inputs),IMAGE_WIDTH*IMAGE_HEIGHT])
        #for i in range(len(inputs)):
    letter=IMGprocess(inputs)
    letter=np.array(letter)
    letter=np.reshape(letter,(-1,IMAGE_WIDTH*IMAGE_HEIGHT))
    data=letter
    prob,index=predict(data,model_path)
    for i in range(len(index)):
        mark+=index[i][0]
        #print(index[i][0])
    return mark

def color_connection(inputs):
    IMAGE_WIDTH,IMAGE_HEIGHT=48,48
    types=['0','1','2','3','4','5','6','7','8',
           '9','A','B','C','D','E','F','G','H',
           'S','T']
    model_path='/Users/Huang/image/utils_dir0'
    #model_path='/Users/Huang/image/trialModelData/pretrained'

    #image_name="imageToSave.jpg"
    #with open("imageToSave.jpg","wb") as fh:
    #    fh.write(image)
    #    fh.close()
    segms=dropfall(inputs)
    data=np.zeros([len(segms),IMAGE_WIDTH*IMAGE_HEIGHT])
    for i in range(len(segms)):
        letter=IMGprocess(segms[i])
        letter=np.array(letter)
        letter=np.reshape(letter,(-1,IMAGE_WIDTH*IMAGE_HEIGHT))
        data[i]=letter
    prob,index=predict(data,model_path)
    mark=10*index[0][0]+1*index[1][0]
    return mark

def detect_connection(image):
    #利用数字的下波形来判断是否是单个数字
    #ret,image=cv2.threshold(image,230,255,cv2.THRESH_BINARY_INV)
    #skeleton=255*np.asarray(morphology.skeletonize(image/255),dtype='uint8')
    #ret,skele_extrac=cv2.threshold(skeleton,230,255,cv2.THRESH_BINARY_INV)
    #ret,skele_extrac=cv2.threshold(image,230,255,cv2.THRESH_BINARY_INV)
    points=[]

    for i in range(image.shape[1]):
        p_top=0
        for j in range(image.shape[0]):
            if image[j,i]==0:
                p_top=max(j,p_top)
        points.append(p_top)
    i=1
    crest=[]
    coords=[]
#    if np.argmax(points)==0:
#        crest.append(1)
#    else:
    '''
    while 1<=i<0.9*len(points):
        #i+=1
        length=0
        if points[i-1]<points[i]>=points[i+1] and points[i]>np.mean(points) and i<0.9*len(points):
            #max(points)*0.8
            i+=1
            length=0
            while points[i-1]>=points[i] and i<len(points)-1:
                i+=1
                length+=1
            if length>0.1*len(points):
            #if length>=5:
                crest.append(length)
        else:
            i+=1
    '''
    while 1<=i<len(points)-1:
        #i+=1
        length=0
        if points[i-1]<points[i]>=points[i+1] and points[i]>np.mean(points) and i<0.9*len(points):
            #max(points)*0.8
            start=points[i]

            i+=1
            length=0
            while points[i-1]>=points[i] and i<len(points)-1:
                i+=1
                length+=1
            end=points[i-1]
            if length>0.1*len(points) and start-end>=5:
            #if length>=5:
                crest.append(length)
                coords.append([i-1-length,i-1])
        else:
            i+=1
    #if points[0]==np.max(points):

            #if i>=0.7*len(points) and len(crest)!=0:
             #   crest.append(length)

        if int(0.9*len(points))<=i<len(points)-1:
            coords.append(i)
            length=0
            while points[i-1]<=points[i] and i<len(points)-1:
                i+=1
                length+=1
            crest.append(length)
            #coords.append(i)
    while 0 in crest:
        crest.pop(crest.index(0))
    #输出波峰的数量，波峰数大于等于2便是连笔多数字
    return (len(crest),[coords[0][0],coords[0][1]])

#%% 测试
#img_path='C:/Users/gd/Desktop/redDigits/38.jpg'

def mark_main(image):

    enh_col = ImageEnhance.Color(image)
    color = 4
    image_colored = enh_col.enhance(color)
    image=np.array(image_colored)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    #hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #hsv=cv2.dilate(hsv,kernel)
    #redLower=np.array([140,10,170])#HSV
    #redHigher=np.array([180,160,250])#HSV
    #redLower=np.array([90,135,120])#LAB
    #redHigher=np.array([210,180,128])#LAB
    redLower=np.array([80,150,20])
    redHigher=np.array([200,220,128])
    mask=cv2.inRange(lab,redLower,redHigher)
    ret,cont,hier=cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points=[]
    res=[]
    temp=[]
    count=0
    for i in cont:
        x,y,w,h=cv2.boundingRect(i)
        if w*h>100:
            points.append([x,y,w,h])
    temp=sorted(points,key=lambda value:value[1])
    while count<len(temp):
        if temp[count][1]-temp[count-1][1]<40 and res:
            x,y,w,h=temp[count]
            tempx,tempy,tempw,temph=res.pop(-1)
            if y>tempy:
                y=tempy
            if h<temph:
                h=temph
            if x>tempx:
                dis=x-tempx-tempw
                x=tempx
                #w=w+tempw
            else:
                dis=tempx-x-w
            if tempw<w+tempw+dis:
                w=w+tempw+dis
            else:
                w=tempw
            res.append([x,y,w,h])
        else:
            res.append(temp[count])
        count+=1
    output=[]
    single=[]
    mark=[]
    for i in range(len(res)):
        x,y,w,h=res[i]
        if h<=20:
            continue
        num=image[y-6:y+h+6,x-6:x+w+6]
        num=cv2.cvtColor(num,cv2.COLOR_BGR2GRAY)
        ret,th=cv2.threshold(num,220,255,cv2.THRESH_BINARY)
        th=cv2.medianBlur(th,3)
        nums=segment_direct(th)
        if len(nums)>=2:
            images=[]
            for j in range(len(nums)):
                x,y,w,h=nums[j]
                if h<20:
                    x0,y0,w0,h0=nums[j-1]
                    if x0>=2 and y0>=2:
                        img=th[y0-2:y0+h0+2,x0-2:x0+w+w0+2]
                    elif x<2 and y>=2:
                        img=th[y0-2:y0+h0+2,x0:x0+w+w0+2]
                    elif x>=2 and y<2:
                        img=th[y0:y0+h0+2,x0-2:x0+w+w0+2]
                    else:
                        img=th[y0:y0+h0+2,x0:x0+w+w0+2]
                    images[j-1]=img

                else:
                    if w*h>100:
                        if x>=2 and y>=2:
                            img=th[y-2:y+h+2,x-2:x+w+2]
                        elif x<2 and y>=2:
                            img=th[y-2:y+h+2,x:x+w+2]
                        elif x>=2 and y<2:
                            img=th[y:y+h+2,x-2:x+w+2]
                        else:
                            img=th[y:y+h+2,x:x+w+2]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                        #img=cv2.dilate(img,kernel,iterations=1)
                        images.append(img)
            if len(images)<2:
                mark.append(color_single(images[0]))
            else:
                points=color_recognize(images)
                mark.append(points)

        #ret,th=cv2.threshold(num,230,255,cv2.THRESH_BINARY_INV)
        #ret,th=cv2.threshold(num,230,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            x,y,w,h=nums[0]
            if x>=2 and y>=2:
                img=th[y-2:y+h+2,x-2:x+w+2]
            elif x<2 and y>=2:
                img=th[y-2:y+h+2,x:x+w+2]
            elif x>=2 and y<2:
                img=th[y:y+h+2,x-2:x+w+2]
            else:
                img=th[y:y+h+2,x:x+w+2]

            #single.append(img)

            num,coords=detect_connection(img)
            if num>=2 :
                try:
                    images=[]
                    start,end=coords
                    images.append(img[:,:start+int((end-start)//2)])
                    images.append(img[:,start+int((end-start)//2):])
                    mark.append(color_recognize(images))
                except:
                    mark.append(color_single(img))
            else:
                mark.append(color_single(img))

    return mark

def RedNum(strs):

    string=strs.split(',')
    output=[]
    d={}
    for j in range(len(string)):
        res=''
        branch=string[j]
        inp=branch.encode('utf-8')
        image_bytes=base64.b64decode(inp)
        image = Image.open(io.BytesIO(image_bytes))
        res=mark_main(image)
        res.append(sum(res))
        d[j+1]=res

    d=json.dumps(res)
    return d


#创建服务器，注册方法
port=8080
#server=TXMLRPCServer(("10.0.0.247",port),SimpleXMLRPCRequestHandler)
server=SimpleXMLRPCServer(("10.0.0.247",port),logRequests=True,allow_none=True)

server.register_function(RedNum)

#server.register_instance(imgrecog)
#server.register_function(recognize)


print("Listening on port %d" %port)
server.serve_forever()
