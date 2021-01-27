import os
from PIL import Image

class transparent_background(object):
    def __init__(self,img):
        self.img = img
    
    def putpixel(self):
        img = Image.open(self.img).convert('RGBA')
        w,h = img.size
        color_0 = (0,0,0,0)
        for i in range(h):
            for j in range(w):
                dot = (j,i)
                color_1 = img.getpixel(dot)
                if color_1 == (255,255,255,255):
                    img.putpixel(dot,color_0)
        return img.convert('RGB')
    
    def rgba(self):
        img = Image.open(self.img).convert('RGBA')
        alpha = Image.new('RGBA',img.size,(0,)*4)
        out = Image.composite(img,alpha,img)
        return out.convert('RGB')


def walk_dir(src):
    def get_one_classlist(classname):
        class_path = os.path.join(src,classname)
        for sample in os.listdir(class_path):
            transparent_background(os.path.join(class_path,sample)).putpixel().save(os.path.join(class_path,sample))
    
    for c in os.listdir(src):
        get_one_classlist(c)        


if __name__ == '__main__':
    walk_dir('D:\\学习\\毕业设计\\Dataset\\classified_dataset')
    #transparent_background('D:\\学习\\毕业设计\\Dataset\\min_crop\\AB_01_005_crop.jpg').putpixel()