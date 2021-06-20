#!/usr/bin/env python
# coding: utf-8

# In[233]:


import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt


# In[234]:


#读取图片
def read_directory(directory_name):
    array_of_img = []
    for filename in os.listdir(r"./"+directory_name): 
        img = cv.imread(directory_name + "/" + filename,cv.IMREAD_GRAYSCALE)
        array_of_img.append(img)
    return array_of_img

train_set=read_directory("train_img")
train_label=read_directory("train_label")
length=len(train_set)
#print(len(train_set),len(test_set))


# In[235]:


# canny算法的评分机制 MPA和MIoU实现
def MPA(pred,label):
    #计算每类各自分类的准确率，再取均值。
    pred=pred/255
    label=label/255
    acc0=np.sum((pred==0) & (label==0))/np.sum(label==0)
    acc1=np.sum((pred==1) & (label==1))/np.sum(label==1)
    return (acc0+acc1)/2

def MIoU(pred,label):
    '''
    单独考虑第i类：
    分子：所有被正确分类为第i类的像素数Pii
    分母：所有标签为第i类的像素数+所有被分类为第i类的像素数-被正确分类为第i类的像素数
    即对每一个类别计算IoU，再对各类求均值。
    '''
    pred=pred/255
    label=label/255
    res=0
    for cls in [0,1]:
        pii=np.sum((pred==cls) & (label==cls))
        iou=pii/(np.sum(label==cls)+np.sum(pred==cls)-pii)
        res+=iou
    return res/2


# In[249]:


a="output_"
b=".jpg"

th1=50
maxMPA=0
maxMIoU=0
bestTH1=0
bestTH2=0
bestKenelSize=0
length=25

#先找到最优阈值 th1,th2
while(th1<501):
    th2=th1+50
    while(th2<th1+501):
        MPA_score=0
        MIoU_score=0
        for i in range(length):
            img=train_set[i]
            label=train_label[i]
            #img = cv.GaussianBlur(img, (3, 3), 0) 
            edges = cv.Canny(img,th1,th2)
            edges=~edges
            score1=MPA(edges,label)
            score2=MIoU(edges,label)
            MPA_score+=score1
            MIoU_score+=score2
        MPA_score/=length
        MIoU_score/=length
        #print(maxMPA)
        if(MIoU_score>maxMPA):
            maxMPA=MIoU_score
            bestTH1=th1
            bestTH2=th2
            print(bestTH1,bestTH2) 
        th2+=50
    th1+=50
    
print(bestTH1,bestTH2) 


# In[242]:


MPA_score=0
length=25

#再找到最优的形态学操作核大小

for i in range(length):
    img=train_set[i]
    label=train_label[i]
    edges = cv.Canny(img,250,500)
    edges=~edges
    score1=MPA(edges,label)
    MPA_score+=score1
    score2=MIoU(edges,label)
    MIoU_score+=score2
MPA_score/=length
MIoU_score/=length
print(MPA_score,MIoU_score)

#实验发现erode核比dilate核大2时效果较好 循环寻找最优kernel size   
for size in range(1,9):
    MPA_score=0
    for i in range(length):#0 黑 白 255 
        img=train_set[i]
        label=train_label[i]
        #img = cv.GaussianBlur(img, (3, 3), 0) 
        edges = cv.Canny(img,250,500)
        edges=~edges
        kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(size+2,size+2))
        edges = cv.erode(edges,kernel1)
        kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(size,size))
        edges = cv.dilate(edges,kernel)
        score1=MPA(edges,label)
        MPA_score+=score1
        score2=MIoU(edges,label)
        MIoU_score+=score2
    MPA_score/=length
    MIoU_score/=length
    #print(MPA_score,MIoU_score)
    if(MIoU_score>maxMPA):
        maxMPA=MIoU_score
        bestkernelsize=size
print(bestkernelsize)  


# MPA_score=0
# for i in range(length):#0 黑 白 255 
#     img=train_set[i]
#     label=train_label[i]
#     edges = cv.Canny(img,200,550)
#     edges=~edges
#     kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(1,1))
#     edges = cv.erode(edges,kernel)
#     score1=MPA(edges,label)
#     MPA_score+=score1
# MPA_score/=length
# print(MPA_score) 

# In[248]:


c="test_"
test_set=read_directory("test_img")
length=len(test_set)
for i in range(5):
    img=test_set[i]
    #img = cv.GaussianBlur(img, (3, 3), 0) 
    edges = cv.Canny(img,250,550)
    edges=~edges
    
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    edges = cv.erode(edges,kernel1)
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    edges = cv.dilate(edges,kernel2)
    cv.imwrite(c+str(i)+b, edges)

