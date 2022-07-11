from PIL import Image #pillow 설치
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '../datasets/cat_dog/train/'
categories = ['cat' , 'dog']
image_w = 64
image_h = 64

pixel = image_w * image_h * 3
x = []
y = []
files = None
for idx, category in enumerate(categories):
    files = glob.glob(img_dir + category + '*.jpg')
    for i, f in enumerate(files):
        try:
            img=Image.open(f)
            img=img.convert('RGB')
            img=img.resize((image_w,image_h))
            data=np.asarray(img)
            x.append(data)
            y.append(idx)#고양이 0,개 1
            if i % 300==0:
                print(category,':',f)
        except:
            print('error:',category,i)
x=np.array(x)
y=np.array(y)
x=x/255
print(x[0])
print(y[0])
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.1)
xy=(x_train,x_test,y_train,y_test)
np.save('../datasets/binary_image_data.npy',xy)