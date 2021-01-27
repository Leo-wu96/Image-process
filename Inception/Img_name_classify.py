# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:12:09 2019

@author: WZX
"""

import os,shutil
import xlrd
import functools


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
    #classify(path,class_dict,save_path)

    return class_dict


path = 'D:\\学习\\毕业设计\\Dataset\\white\\img3'
save_path = 'D:\\学习\\毕业设计\\Dataset\\white\\classifed_dataset'
xlsx_file = 'D:\\学习\\毕业设计\\Dataset\\list.xlsx'





if __name__ == '__main__':
    class_dict = Img_name_classify(path,save_path,xlsx_file)


