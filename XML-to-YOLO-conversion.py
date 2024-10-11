# Description: This script is used to convert the XML files to YOLO format.
# Author: wang
# Date: 2024-10-09
# Reference: https://blog.csdn.net/weixin_45679938/article/details/118803745

import xml.etree.ElementTree as ET #解析xml文件
import pickle #序列化和反序列化
import os #操作系统接口

from os import listdir, getcwd #
from os.path import join #连接两个或更多的路径名组件

sets = ['train', 'val', 'test'] #训练集、验证集、测试集
classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper'] #类别, 6类

def convert(size, box): #将box的坐标转换为yolo需要的坐标
    dw = 1./size[0]  #图片的宽度
    dh = 1./size[1]  #图片的高度
    x = (box[0] + box[1])/2.0   #box的中心点的x坐标
    y = (box[2] + box[3])/2.0   #box的中心点的y坐标
    w = box[1] - box[0]         #box的宽度
    h = box[3] - box[2]         #box的高度
    x = x*dw 
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h) #返回yolo需要的坐标

def convert_annotation(image_id): #将xml文件转换为yolo需要的label文件, image_id是xml文件的名字
    in_file=open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/Annotations/%s.xml'%(image_id),encoding='utf-8') #打开xml文件
    out_file=open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/labels/%s.txt'%(image_id), 'w',encoding='utf-8') #打开txt文件
    tree=ET.parse(in_file) #解析xml文件
    root = tree.getroot() #获取根节点
    size = root.find('size') #获取size节点
    if size != None:
        w=int(size.find('width').text) #获取图片的宽度
        h=int(size.find('height').text) #获取图片的高度
        for obj in root.iter('object'): #遍历object节点
            difficult = obj.find('difficult').text #获取difficult节点
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1: #如果类别不在classes中或者difficult=1,则跳过
                continue
            cls_id = classes.index(cls) #获取类别的索引
            xmlbox = obj.find('bndbox') #获取bndbox节点
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)) #获取box的坐标
            print(image_id,cls,b)
            bb = convert((w,h), b) #将box的坐标转换为yolo需要的坐标
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n') #写入txt文件


wd = getcwd() #获取当前工作目录
print(wd) #打印当前工作目录

for image_set in sets: #遍历训练集、验证集、测试集
    if not os.path.exists('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/labels/'):
        os.makedirs('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/labels/')
    
    image_ids = open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/ImageSets/%s.txt'%(image_set)).read().strip().split() #读取txt文件
    list_file = open('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/%s.txt'%(image_set), 'w') #打开txt文件
    

    for image_id in image_ids: #遍历所有的xml文件
        list_file.write('/mnt/smbmount/1_industry_dataset/PCB_train_data/data/images/%s.jpg\n'%(image_id))
        convert_annotation(image_id) #将xml文件转换为yolo需要的label文件
        
    list_file.close() #关闭txt文件 