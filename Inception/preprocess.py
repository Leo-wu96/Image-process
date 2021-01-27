# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:07:40 2019

@author: WZX

This code is used for preprocessing the images

"""

from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np 

def file_rename(path,item,pre): #更改文件名
    old_name = os.path.join(path + '\\' + pre,item)
    #filename = os.path.splitext(item)[0]
    filetype = os.path.splitext(item)[1]
    new_name = os.path.join(path + '\\' + pre,pre + '_' + item + filetype)
    os.rename(old_name,new_name)
        
def batch_rename(path):  #修改目录中所有JPG 图像的文件名列表
    path_list = os.listdir(path)
    for p in path_list:
        filelist = os.listdir(os.path.join(path,p))
        pre = os.path.basename(p)
        for item in filelist:
            file_rename(path,item,pre)
    #return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.JPG')]



def errorFilename_rename(path):
    def rename(path,item):    #item是filename
        old_name = os.path.join(path,item)
        filename = os.path.splitext(item)[0]
        filetype = os.path.splitext(item)[1]
        
        if ' ' in filename:
            filename = filename.replace(' ','_')
        if '-' in filename:
            filename = filename.replace('-','_')
        if '.' in filename:
            filename = filename.replace('.','_')
        if '&' in filename:
            filename = filename.replace('&','_')
        new_name = os.path.join(path,filename + filetype)
        os.rename(old_name,new_name)
        
    filelist = os.listdir(path)
    for item in filelist:
        rename(path,item)
            
    
#数据集路径
path='D:\\学习\\毕业设计\\Dataset\\process_img' 



if __name__ == '__main__':
     errorFilename_rename(path)

