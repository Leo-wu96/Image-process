# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:08:25 2019

File:Data_Augmentation.py

@author: WZX

"""
import cv2
import numpy as np
import os
import random
from PIL import Image,ImageEnhance


class Data_Augmentation(object):
    """Image process include rotate/bright/crop/shift/resize."""
    
    def __init__(self,path,savePath):
        self.img_list = []
        for i in os.listdir(path):
            self.img_list.append(os.path.join(path,i))
        
            
        self.path = path
        self.savepath = savePath
        
        
        
        #self.crop()
        #self.random_crop()
        #self.random_rotation()
        #self.flip()
        #self.shift()
        #self.resize()
        #self.color()
        #self.bright()
        #self.contrast()
        #self.sharp()
    
    
    
    
    
    
    def makedir(self,name):
        Dir = os.path.join(self.savepath,name)
        if not os.path.exists(Dir):
            os.makedirs(Dir)
            
        return os.path.join(Dir,'')
            
            
    
    
    
    def get_one_classlist(path,classname):
        sample_list = []
        class_path = os.path.join(path,classname)
        for sample in os.listdir(class_path):
            sample_list.append(os.path(class_path,sample))
        
        return sample_list
    
    
    
    
    
    
    def cv_imread(path):
        img = cv2.imdecode(np.fromfile(path,dtype = np.uint8),-1)
        return img
    
    
    
    
    
    
    
    
    
    
    def __iter__(self):
        for i in range(len(self.img_list)):
            yield Image.open(self.img_list[i],mode = 'r')
    
    
    
    
    
    def putpixel(self):
        """背景换成黑色"""
        path = self.makedir('black')
        count = 0
        for img in self:
            filename = os.path.splitext(os.path.basename(self.img_list[count]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[count]))[1]
            
            
            w,h = img.size
            color_0 = (0,0,0)
            for i in range(h):
                for j in range(w):
                    dot = (j,i)
                    color_1 = img.getpixel(dot)
                    if color_1 == (255,255,255):
                        img.putpixel(dot,color_0)
            img.save(path+filename+'_black'+filetype,quality = 95)
            count += 1
    
    
    
    def crop(self):
        """裁剪成2990*2990图片大小"""
        path = self.makedir('crop')
        i = 0
        for img in self:
            width = img.size[0]
            height = img.size[1]
            #box = (width//2-1495, height//2-1495, width//2+1495, height//2+1495)
            box = (width//2-1200, height//2-1200, width//2+1200, height//2+1200)
            crop_img = img.crop(box)
            
            filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
            crop_img.save(path + filename+'_crop'+ filetype,quality = 95)
            i += 1
            
    
    
     
    
    
    def random_crop(self):
        """
        对图像随意剪切,考虑到图像大小范围,使用一个随机窗口进行截图
        image: 图像image
        return: 剪切之后的图像
        """
        path = self.makedir('random_crop')
        i = 0
        for img in self:
            start = random.randrange(100,500)
            end = random.randrange(2500,2900)
            box = (start, start, end, end)
            
            filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
            img.crop(box).save(path + filename +'_random_crop'+ filetype,quality = 95)
            i += 1
    
    
    
    
    
    
    
    
    
    
    
    def random_rotation(self):
        """随机旋转图像，并填充白色背景"""
        path = self.makedir('random_rotation')
        i = 0
        for img in self:
            angle = random.randrange(1,360)
            
            #转换为alpha层
            #img = img.convert('RGBA')
            rotate_img = img.resize((299,299)).rotate(angle)  #expand = 1为放大图像 = 0不变
            
            #white_channal = Image.new('RGBA',rotate_img.size,(255,)*4)
            
            #rotate_img_bound = Image.composite(rotate_img,white_channal,rotate_img)
            
            filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
            rotate_img.save(path + filename +'_random_rotation'+ filetype,quality = 95)
            i += 1
        
    
    
    
    
    
    
    
    
    
    def flip(self):
        name = ['flip_v','flip_h']
        for j,n in enumerate(name):
            i = 0
            path = self.makedir(n)
            for img in self:
                filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
                filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
                
                #img.resize((299,299)).transpose(random.randrange(0,2)).save(path + filename + n + filetype,quality = 95)
                img.resize((299,299)).transpose(j).save(path + filename + n + filetype,quality = 95)
                #img.resize((299,299)).transpose(1).save(path + filename + n + filetype,quality = 95)
                #img.resize((299,299)).transpose(0).transpose(1).save(path + filename +'_vertrical_horizontal_flip'+ filetype,quality = 95)
                i += 1
    
    
    
   
    
    
    
    
    
    def shift(self):
        path = self.makedir('shift')
        count = 0
        
        for img in self:
            filename = os.path.splitext(os.path.basename(self.img_list[count]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[count]))[1]
            
            
            img_array = np.array(img)
            
            s = random.randrange(100,1000)
            #Shift right
            for i in range(img_array.shape[1]-1,0,-1):
                for j in range(img_array.shape[0]):
                    if i >= s:
                        img_array[j][i] = img_array[j][i-s]
                    else:
                        img_array[j][i] = 255
            Image.fromarray(img_array.astype('uint8')).convert('RGB').save(path + filename +'_Shift_right'+ filetype,quality = 95)
            
            s = random.randrange(100,1000)
            #Shift left
            for i in range(img_array.shape[1]):
                for j in range(img_array.shape[0]):
                    if i + s < img_array.shape[1]:
                        img_array[j][i] = img_array[j][i + s]
                    else:
                        img_array[j][i] = 255
            Image.fromarray(img_array.astype('uint8')).convert('RGB').save(path + filename +'_Shift_left'+ filetype,quality = 95)
            
            
            
            s = random.randrange(100,1000)
            #Shift up
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    if i + s < img_array.shape[0]:
                        img_array[i][j] = img_array[i+s][j]
                    else:
                        img_array[i][j] = 255
            Image.fromarray(img_array.astype('uint8')).convert('RGB').save(path + filename +'_Shift_up'+ filetype,quality = 95)


            s = random.randrange(100,1000)
            #Shift down
            for i in range(img_array.shape[0]-1,0,-1):
                for j in range(img_array.shape[1]):
                    if i >= s:
                        img_array[i][j] = img_array[i-s][j]
                    else:
                        img_array[i][j] = 255
            Image.fromarray(img_array.astype('uint8')).convert('RGB').save(path + filename +'_Shift_down'+ filetype,quality = 95)
            
            
            count += 1
            
    
    
    
    
    
    
    
    
    def resize(self):
        path = self.makedir('resize')
        i = 0
        for img in self:
            filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
            
            
            img.resize((299,299)).save(path + filename +'_resize_299'+ filetype,quality = 95)
            #img.resize((int(img.size[0]*1.5),int(img.size[1]*1.5))).save(path + filename +'_resize_1_5'+ filetype,quality = 95)
            
            
            i += 1
  
    
    
    
    
    
    
    
    
    
    
    #颜色增强
    def color(self):
        
        color = [0.5,1.25,1.5]
        
        for c in color:
            i = 0
            path = self.makedir('color_'+ str(c).replace('.','_'))
            for img in self:
                filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
                filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
                enh_col = ImageEnhance.Color(img.resize((299,299)))
                image_colored = enh_col.enhance(c)
                image_colored.save(path + filename +'_color_'+ str(c).replace('.','_')+ filetype,quality = 95)
                i += 1
   
    
    
    
    
    
    
    
    
    
    #亮度增强
    def bright(self):
        path = self.makedir('bright')
        i = 0
        for img in self:
            filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
            
            
            enh_bri = ImageEnhance.Brightness(img.resize((299,299)))
            brightness = 1.5
            image_brightened = enh_bri.enhance(brightness)
            #image_brightened.resize((299,299)).save(path + filename+'_bright_1_5'+ filetype,quality = 95)
            image_brightened.save(path + filename+'_bright_1_5'+ filetype,quality = 95)
            i += 1
 
    
    
   
    
    
    
    
    #对比度增强
    def contrast(self):
        path = self.makedir('contrast')
        i = 0
        for img in self:
            filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
            
            
            enh_con = ImageEnhance.Contrast(img)
            contrast = 1.2
            image_contrasted = enh_con.enhance(contrast.resize((299,299)))
            #image_contrasted.resize((299,299)).save(path + filename +'_contrast_1_2'+ filetype,quality = 95)
            image_contrasted.save(path + filename +'_contrast_1_2'+ filetype,quality = 95)
            i += 1





     #锐度增强       
    def sharp(self):
        path = self.makedir('sharp')
        i = 0
        sharpness = [1.25,1.5]
        for img in self:
            filename = os.path.splitext(os.path.basename(self.img_list[i]))[0]
            filetype = os.path.splitext(os.path.basename(self.img_list[i]))[1]
            
            
            
            enh_sha =  ImageEnhance.Sharpness(img)
            for s in sharpness:
                image_sharped = enh_sha.enhance(s)
                image_sharped.resize((299,299)).save(path + filename+'_sharp_'+ str(s).replace('.','_')+ filetype,quality = 95)
            i += 1
            
            
            
            


def extra():
    """对已分类的图像进行附加的操作。"""
    def get_one_classlist(path,classname):
            sample_list = []
            class_path = os.path.join(path,classname)
            for sample in os.listdir(class_path):
                sample_list.append(os.path.join(class_path,sample))
            
            return sample_list
       
        
        
    
    def random_rotation(img_path):
        """随机旋转图像，并填充白色背景"""
        angle = random.randrange(1,360)
        img = Image.open(img_path,mode = 'r')
        
        
        #转换为alpha层
        img = img.convert('RGBA')
        rotate_img = img.rotate(angle)  #expand = 1为放大图像 = 0不变
        
        white_channal = Image.new('RGBA',rotate_img.size,(255,)*4)
        
        rotate_img_bound = Image.composite(rotate_img,white_channal,rotate_img)
        
    
        rotate_img_bound.convert('RGB').resize((299,299)).save(img_path,quality = 95)
        
    path = 'D:\\学习\\毕业设计\\Dataset\\classifed_dataset'
    
    for c in os.listdir(path):
        sample_list = get_one_classlist(path,c)
        for p in sample_list:
            random_rotation(p)





class augmentation(object):
    """1张图片增广10张"""
    def __init__(self,path,savepath):
        self.class_list = []
        for i in os.listdir(path):
            self.class_list.append(os.path.join(path,i))
        
            
        self.path = path
        self.savepath = savepath
        
        for class_path in self:
            sample_list = self.get_one_classlist(class_path)
            for img_path in sample_list:
                self.random_rotation(img_path)
                self.flip(img_path)
                self.resize(img_path)
                self.color(img_path)
                self.bright(img_path)
                self.contrast(img_path)
                self.sharp(img_path)
    



    
    def get_one_classlist(self,class_path):
        sample_list = []
        for sample in os.listdir(class_path):
            sample_list.append(os.path.join((class_path,sample)))
        
        return sample_list
    
    
    
    
    
    
    def __iter__(self):
        for i in len(self.class_list):
            yield self.class_list[i]
            
   
    
    
    
    def random_rotation(self,img_path):
        """随机旋转图像"""
        angle = random.randrange(1,360)
        

        rotate_img = Image.open(img_path).resize((299,299)).rotate(angle)  #expand = 1为放大图像 = 0不变
        
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
        rotate_img.save(os.path.split(img_path) + filename +'_random_rotation'+ filetype,quality = 95)
            
    
    
    def flip(self,img_path):
        name = ['flip_v','flip_h']
        img = Image.open(img_path)
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
            
            
            
        for j,n in enumerate(name):
            #img.resize((299,299)).transpose(random.randrange(0,2)).save(path + filename + n + filetype,quality = 95)
            img.resize((299,299)).transpose(j).save(os.path.split(img_path) + filename + n + filetype,quality = 95)
            #img.resize((299,299)).transpose(1).save(path + filename + n + filetype,quality = 95)
            #img.resize((299,299)).transpose(0).transpose(1).save(path + filename +'_vertrical_horizontal_flip'+ filetype,quality = 95)
            
            
            
            
    def resize(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
            
            
        img.resize((299,299)).save(os.path.split(img_path) + filename +'_resize_299'+ filetype,quality = 95)
        
        
        
        
    #颜色增强
    def color(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
        
        
        color = [0.5,1.5]
        
        for c in color:
            enh_col = ImageEnhance.Color(img)
            image_colored = enh_col.enhance(c)
            image_colored.resize((299,299)).save(os.path.split(img_path) + filename +'_color_'+ str(c).replace('.','_')+ filetype,quality = 95)
   
    
    
    
    
    
    
    
    
    
    #亮度增强
    def bright(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
            
            
        enh_bri = ImageEnhance.Brightness(img)
        brightness = 1.5
        image_brightened = enh_bri.enhance(brightness)
        image_brightened.resize((299,299)).save(os.path.split(img_path) + filename+'_bright_1_5'+ filetype,quality = 95)
 
    
    
   
    
    
    
    
    #对比度增强
    def contrast(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
            
            
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.2
        image_contrasted = enh_con.enhance(contrast)
        image_contrasted.resize((299,299)).save(os.path.split(img_path) + filename +'_contrast_1_2'+ filetype,quality = 95)






    #锐度增强 
    def sharp(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
        
        
        sharpness = [1.25,1.5]

        enh_sha = ImageEnhance.Sharpness(img)
        for s in sharpness:
            image_sharped = enh_sha.enhance(s)
            image_sharped.resize((299,299)).save(os.path.split(img_path) + filename+'_sharp_'+ str(s).replace('.','_')+ filetype,quality = 95)
     
        
        

Path = 'D:\\学习\\毕业设计\\Dataset\\classifed_img'         
savePath = 'D:\\学习\\毕业设计\\Dataset'








if __name__ == '__main__':
    Data_Augmentation(Path,savePath)  
    #extra()




