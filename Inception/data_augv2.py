# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:16:54 2019

@author: WZX
"""
import os,shutil
import xlrd
import functools
import numpy as np
import cv2
import random
from PIL import Image,ImageEnhance


def Img_name_classify(path,save_path,xlsx_file):
    """将不同类别图片归在不同文件夹"""

    def repath(filepath,newdir):
        """更改文件路径
        filepath:文件路径
        newdir:新路径
        """
        _,fname = os.path.split(filepath)
        shutil.copy(filepath,os.path.join(newdir,fname))





    def class_sorted_dict(xlsx_file):
        """输入Excel文件,对其单元格排序.
            输出排序好的单元格元祖"""

        def readExcel(file):
            sheet = xlrd.open_workbook(filename=file).sheet_by_index(0)#打开文件,通过索引获取表格
            print(sheet.name,sheet.nrows,sheet.ncols)

            #className = sheet.row_values(0)   #获取行内容
            className = sheet.col_values(0)   #获取列内容
            sampleNum = sheet.col_values(1)   #获取列内容

            return className,sampleNum





        def rename(filename):
            """替换字符串中文字符"""
            if ' ' in filename:
                filename = filename.replace(' ','_')
            if '-' in filename:
                filename = filename.replace('-','_')
            if '.' in filename:
                filename = filename.replace('.','_')
            if '&' in filename:
                filename = filename.replace('&','_')

            return filename





        def cmp_ignore_case(s1,s2):
            """输入为字符串"""
            s1 = s1[0].lower()
            s2 = s2[0].lower()
            if s1 < s2:
                return -1
            if s1 > s2:
                return 1

            return 0


        className,sampleNum = readExcel(xlsx_file)
        dict_class = {}
        for i in range(len(className)):
            dict_class[className[i]] = sampleNum[i]

        for key in list(dict_class.keys()):
            dict_class[rename(key)] = dict_class.pop(key)

        new_dict = sorted(dict_class.items(),key = functools.cmp_to_key(cmp_ignore_case))
        #print(new_dict)

        return new_dict




    def file_list(dirs):
        """
        for d in os.listdir(dirs):
            if os.path.isdir(os.path.join(dirs,d)):
                file_list(os.path.join(dirs,d))
            else:
                sorted([os.path.join(dirs,f) for f in os.listdir(dirs)],key = str.lower)
                #classify(files,class_dict,save_path)
        """
        pass



    def classify(path,class_dict,save_path):
        files = sorted([os.path.join(path,f) for f in os.listdir(path)],key = str.lower)
        for c in class_dict:
            if not os.path.exists(os.path.join(save_path,c[0])):
                os.makedirs(os.path.join(save_path,c[0]))
            for i in range(int(c[1])):
                for j in range(8):
                    repath(files.pop(0),os.path.join(save_path,c[0]))




    class_dict = class_sorted_dict(xlsx_file)
    classify(path,class_dict,save_path)











class augmentation(object):
    """1张图片增广10张"""
    def __init__(self,path):
        """随机切割酒塞"""

        self.path = path



        self.class_list = []
        for i in os.listdir(self.path):
            self.class_list.append(os.path.join(self.path,i))



        for class_path in self:
            sample_list = self.get_one_classlist(class_path)
            for img_path in sample_list:
                self.random_crop(img_path)
                #self.random_rotation(img_path)
                #self.flip(img_path)
                #self.resize(img_path)
                #self.color(img_path)
                #self.bright(img_path)
                #self.contrast(img_path)
                #self.sharp(img_path)



    def get_one_classlist(self,class_path):
        sample_list = []
        for sample in os.listdir(class_path):
            sample_list.append(os.path.join(class_path,sample))

        return sample_list











    def __iter__(self):
        for i in range(len(self.class_list)):
            yield self.class_list[i]









    def contours_cut(self,img_path,a):
        """根据轮廓切割"""

        img = cv2.imread(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]


        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



        x, y, w, h = cv2.boundingRect(contours[1])


        newimage=img[y+a:y+h-a,x+a:x+w-a] # 先用y确定高，再用x确定宽


        imgdir=('dataset/'+ str(a)+"/")
        if not os.path.isdir(imgdir):
            os.makedirs(imgdir)
        cv2.imwrite(imgdir + filename + filetype, cv2.resize(newimage,(299,299)))




    def random_crop(self,img_path):
        img = Image.open(img_path)
        width = img.size[0]
        height = img.size[1]

        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]

        i = 0
        a = 299*299*3
        while True:
            x = random.randrange(150,width-299)
            y = random.randrange(150,height-299)
            box = (x, y, x+299, y+299)
            crop = np.array(img.crop(box))
            if np.sum(crop==255)/a < 0.1:
                img.crop(box).save(os.path.split(img_path)[0]+'\\' + filename +'_'+str(i+1)+ filetype,quality = 95)
                i += 1
            if i == 20:
                os.remove(img_path)
                break

    def random_rotation(self,img_path):
        """随机旋转图像"""
        angle = random.randrange(1,360)


        rotate_img = Image.open(img_path).resize((299,299)).rotate(angle)  #expand = 1为放大图像 = 0不变


        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]
        rotate_img.save(os.path.split(img_path)[0]+'\\' + filename +'_random_rotation'+ filetype,quality = 95)



    def flip(self,img_path):
        name = ['flip_v','flip_h']
        img = Image.open(img_path)

        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]



        for j,n in enumerate(name):
            #img.resize((299,299)).transpose(random.randrange(0,2)).save(path + filename + n + filetype,quality = 95)
            img.resize((299,299)).transpose(j).save(os.path.split(img_path)[0]+'\\' + filename + n + filetype,quality = 95)
            #img.resize((299,299)).transpose(1).save(path + filename + n + filetype,quality = 95)
            #img.resize((299,299)).transpose(0).transpose(1).save(path + filename +'_vertrical_horizontal_flip'+ filetype,quality = 95)




    def resize(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]


        img.resize((299,299)).save(os.path.split(img_path)[0] +'\\'+ filename +'_resize_299'+ filetype,quality = 95)




    #颜色增强
    def color(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]


        color = [0.5,1.5]

        for c in color:
            enh_col = ImageEnhance.Color(img)
            image_colored = enh_col.enhance(c)
            image_colored.resize((299,299)).save(os.path.split(img_path)[0] +'\\'+ filename +'_color_'+ str(c).replace('.','_')+ filetype,quality = 95)










    #亮度增强
    def bright(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]


        enh_bri = ImageEnhance.Brightness(img)
        brightness = 1.5
        image_brightened = enh_bri.enhance(brightness)
        image_brightened.resize((299,299)).save(os.path.split(img_path)[0] +'\\'+ filename+'_bright_1_5'+ filetype,quality = 95)








    #对比度增强
    def contrast(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]


        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.2
        image_contrasted = enh_con.enhance(contrast)
        image_contrasted.resize((299,299)).save(os.path.split(img_path)[0] +'\\'+ filename +'_contrast_1_2'+ filetype,quality = 95)






    #锐度增强
    def sharp(self,img_path):
        img = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filetype = os.path.splitext(os.path.basename(img_path))[1]


        sharpness = [1.25,1.5]

        enh_sha = ImageEnhance.Sharpness(img)
        for s in sharpness:
            image_sharped = enh_sha.enhance(s)
            image_sharped.resize((299,299)).save(os.path.split(img_path)[0]+'\\' + filename+'_sharp_'+ str(s).replace('.','_')+ filetype,quality = 95)






path = 'D:\\学习\\毕业设计\\Dataset\\compare'
#xlsx_file = 'D:\\学习\\毕业设计\\Dataset\\list.xlsx'






if __name__ == '__main__':
    augmentation(path)


