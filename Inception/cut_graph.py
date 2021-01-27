# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:44:37 2019

@author: WZX
"""

import os
import numpy as np
import cv2
import random
from PIL import Image,ImageEnhance

def preprocess_img(img_path):
    def black(img):
        """将背景置黑"""

        #获取mask
        #lower_white = np.array([254,254,254])
        #upper_white = np.array([255,255,255])
        #mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_white, upper_white)


        #腐蚀膨胀
        #mask = cv2.erode(mask,None,iterations=1)
        #mask = cv2.dilate(mask,None,iterations=1)
        _, mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV)




        image = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)

        return image





    def contours_cut(img):
        """根据轮廓切割"""
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            newimage = img[y-100:y+h+100,x-100:x+w+100] # 先用y确定高，再用x确定宽

            if newimage.shape[0] > 500 and newimage.shape[0] < 2000 and newimage.shape[1] > 500 and newimage.shape[1] < 2000:
                return newimage



    img = cv2.imread(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    filetype = os.path.splitext(os.path.basename(img_path))[1]


    imgdir=("img2/")
    if not os.path.isdir(imgdir):
        os.makedirs(imgdir)


    cv2.imwrite(imgdir+ filename+'_100' + filetype, contours_cut(img))


def putpixel(img_path):
    """背景换成黑色"""
    img = Image.open(img_path)


    w,h = img.size
    color_0 = (0,0,0)
    for i in range(h):
        for j in range(w):
            dot = (j,i)
            color_1 = img.getpixel(dot)
            if color_1 == (255,255,255) or color_1 == (254,254,254):
                img.putpixel(dot,color_0)
    img.save(img_path,quality = 95)

def random_rotation(img_path):
  """随机旋转图像"""
  angle = random.randrange(1,360)
  img = Image.open(img_path).convert('RGBA').rotate(angle)
  white_channal = Image.new('RGBA',img.size,(255,)*4)
  rotate_img = Image.composite(img,white_channal,img)
  #rotate_img = Image.open(img_path).rotate(angle)  #expand = 1为放大图像 = 0不变

  rotate_img.convert('RGB').save(img_path,quality = 95)

def sharp(img_path):
  img = Image.open(img_path)

  enh_sha = ImageEnhance.Sharpness(img)
  image_sharped = enh_sha.enhance(1.2)
  image_sharped.save(img_path,quality = 95)


def flip(img_path):
  img = Image.open(img_path)
  img.transpose(random.randrange(0,2)).save(img_path,quality = 95)



"""
for img_path in os.listdir('process_img'):
    preprocess_img(os.path.join('process_img',img_path))
"""


for img_path in os.listdir('img1'):
    #putpixel(os.path.join('img_dir_1' , img_path))
    #sharp(os.path.join('img_dir_2' , img_path))


    flip(os.path.join('img1' , img_path))
    random_rotation(os.path.join('img1' , img_path))
