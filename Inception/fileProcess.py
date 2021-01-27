# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:29:09 2019

@author: WZX
"""
import os
import xlrd 

def readExcel(file):
    sheet = xlrd.open_workbook(filename=file).sheet_by_index(0)#打开文件,通过索引获取表格
    print(sheet.name,sheet.nrows,sheet.ncols)
    
    #className = sheet.row_values(0)   #获取行内容
    className = sheet.col_values(0)   #获取列内容
    sampleNum = sheet.col_values(1)   #获取列内容
    
    return className,sampleNum
    
def file_rename(path,item,i): #简单更改文件名
    old_name = os.path.join(path,item)
    #filename = os.path.splitext(item)[0]
    filetype = os.path.splitext(item)[1]
    new_name = os.path.join(path,'IMG_'+str(i).zfill(4)+filetype)
    os.rename(old_name,new_name)
        
def fileRename(path,item,i,classNum,className,sample): #按类别/样本更改文件名
    old_name = os.path.join(path,item)
    filetype = os.path.splitext(item)[1]
    new_name = os.path.join(path,className+'_'+str(sample).zfill(2)+'_'+str(i).zfill(3)+filetype)
    os.rename(old_name,new_name)
    

    
def get_imlist(path):  #返回目录中所有JPG 图像的文件名列表
    filelist = os.listdir(path)
    start = 0
    for cla in range(len(sampleNum)):
            for num in range(int(sampleNum[cla])):
                i = 1
                while i < 5:
                    fileRename(path,filelist[start],i,cla+1,className[cla],num+1)
                    i += 1
                    start += 1
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.JPG')]

path='D:\\学习\\毕业设计\\Dataset\\process_img' 
excelFile = 'D:\\学习\\毕业设计\\Dataset\\list.xlsx'

className,sampleNum = readExcel(excelFile)
_ = get_imlist(path)