# Description: Divide the training set into training set and validation set
# Author: wangwei83
# Date: 2024-10-09
# Reference: 
import os 
import random

trainval_percent = 0.9
train_percent = 0.9

xmlfilepath='/mnt/smbmount/1_industry_dataset/PCB_train_data/data/Annotations' #xml文件路径 445xml files
txtsavepath='/mnt/smbmount/1_industry_dataset/PCB_train_data/data/ImageSets' #txt文件保存路径 trainval.txt test.txt train.txt val.txt
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)  #xml文件个数
indices=range(num)    #生成一个0到num-1的列表
tv=int(num*trainval_percent)  #tv=400
tr=int(tv*train_percent)   #tr=360
trainval= random.sample(indices,tv)  #随机选取tv个数，trainval是一个列表
train=random.sample(trainval,tr)  #随机选取tr个数，train是一个列表

ftrainval=open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/ImageSets/trainval.txt','w')  #打开trainval.txt文件,写入,如果不存在则创建
ftest=open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/ImageSets/test.txt','w')  #打开test.txt文件,写入,如果不存在则创建
ftrain=open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/ImageSets/train.txt','w')  #打开train.txt文件,写入,如果不存在则创建
fval=open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/ImageSets/val.txt','w')  #打开val.txt文件,写入,如果不存在则创建

for i  in indices: #遍历所有xml文件
    name=total_xml[i][:-4]+'\n'  #去掉.xml后缀
    if i in trainval:
        ftrainval.write(name)  #写入trainval.txt文件
        if i in train:
            ftrain.write(name)  #写入train.txt文件
        else:
            fval.write(name)  #写入val.txt文件
    else:
        ftest.write(name)  #写入test.txt文件

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()