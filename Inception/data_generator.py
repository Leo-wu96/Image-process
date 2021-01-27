"""
File:data_generator.py
"""


import os
import random
import numpy as np
from PIL import Image




class dataset_split(object):
    """将数据集分成训练集、验证集和测试集
        src:源数据集
        return:csv文件记录数据路径与标签."""

    def __init__(self,src):

        self.src = src
        #self.seed = 100
        self.build_dataset()







    def get_one_classlist(self,classname):
        #random.seed(self.seed)
        sample_list = []
        class_path = os.path.join(self.src,classname)
        for sample in os.listdir(class_path):
            sample_list.append(os.path.join(class_path,sample))

        random.shuffle(sample_list)

        return sample_list








    def build_dataset(self):
        class_num,sample_num = 0,0
        test,validset,trainset = [],[],[]
        label = 0
        #random.seed(24)
        test_class = ['BENJAMIN_LEROUX','Giiaid','GNARIY_HEAD','LALLIER','Two_Hands_Wines']

        for c in os.listdir(self.src):
            class_num += 1
            sample_list = self.get_one_classlist(c)
            sample_num += len(sample_list)

            if c in test_class:
                test.append(sample_list)
            #1:9比例分验证集和训练集
            validset += zip(sample_list[:int(len(sample_list)*0.1)],[label]*len(sample_list[:int(len(sample_list)*0.1)]))
            trainset += zip(sample_list[int(len(sample_list)*0.1):],[label]*len(sample_list[int(len(sample_list)*0.1):]))

            label += 1

        testset = []
        for i,classes in enumerate(test):
            for k in range(10000):
                same_pair = random.sample(classes,2)
                testset.append(same_pair[0],same_pair[1],1)

            for k in range(10000):
                j = i
                while j == i:
                    j = random.randint(0, len(classes)-1)
                testset.append((random.choice(test[i]), random.choice(test[j]), 0))

        random.shuffle(testset)
        random.shuffle(validset)
        random.shuffle(trainset)





        print('\tclass\tsample')
        print('total:\t%6d\t%7d' % (class_num, sample_num))
        print('test:\t%6d\t%7d' % (len(test), len(testset)))
        print('valid:\t%6d\t%7d' % (label, len(validset)))
        print('train:\t%6d\t%7d' % (label, len(trainset)))


        self.set_to_csv_file(testset, 'testset.csv')
        self.set_to_csv_file(validset, 'validset.csv')
        self.set_to_csv_file(trainset, 'trainset.csv')






    def set_to_csv_file(self,dataset,file_name):
        with open(file_name, "w") as f:
            for item in dataset:
                print(" ".join(map(str, item)), file = f)





    def vectorize_imgs(img_path):  #图片变成向量，return arr_img
        with Image.open(img_path) as img:
            arr_img = np.asarray(img, dtype='float32')
            return arr_img




    def read_csv_file(self,csv_file):
        x, y = [], []
        with open(csv_file, "r") as f:
            for line in f.readlines():
                path, label = line.strip().split()
                x.append(self.vectorize_imgs(path))
                y.append(int(label))

        #np.asarray()强制转换成向量并且为指定的数据类型
        return np.asarray(x, dtype='float32'), np.asarray(y, dtype='int32')







#src = 'D:\学习\毕业设计\Dataset\classifed_dataset'
#dataset_split(src)










class generator():
    """批量数据生成器"""

    def __init__(self,csv_file,batch_size):
        self.batch_size = batch_size  #批量数据大小
        self.csv_file = csv_file
        self.index = 0   #指针开始的位置



        with open(self.csv_file, "r") as f:
            self.data_list = [line.strip() for line in f.readlines()]   #读入csv数据列表

        self.count = len(self.data_list)




    def vectorize_imgs(self,img_path):  #图片变成向量，return arr_img
        with Image.open(img_path) as img:
            arr_img = np.asarray(img, dtype='float32')
            return arr_img/255.






    def next(self):
        datas,labels = [],[]
        random.seed(7)


        while True:
            self.index += 1
            #print(self.index)
            self.index = self.index % self.count
            if self.index == 0:
                random.shuffle(self.data_list)
            path, label = self.data_list[self.index].split()
            datas.append(self.vectorize_imgs(path))
            labels.append(int(label))

            if len(datas) == self.batch_size:
                break


        datas = np.asarray(datas,dtype = 'float32')

        return datas, np.asarray(labels,dtype = 'int32')


    def __next__(self):
        return self.next()







