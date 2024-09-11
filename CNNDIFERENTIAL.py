import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def apply_differential_privacy(data, epsilon=1.0):
    sensitivity = 1.0
    scale = sensitivity / epsilon
    noisy_data = data + np.random.laplace(0, scale, data.shape)
    return noisy_data

df = pd.read_csv('data/Car_Hacking_5%.csv')
print(df.Label.value_counts())  

numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer()
df[numeric_features] = scaler.fit_transform(df[numeric_features])


epsilon = 0.5  
df[numeric_features] = df[numeric_features].apply(lambda x: apply_differential_privacy(x, epsilon))

df0 = df[df['Label'] == 'R'].drop(['Label'], axis=1)
df1 = df[df['Label'] == 'RPM'].drop(['Label'], axis=1)
df2 = df[df['Label'] == 'gear'].drop(['Label'], axis=1)
df3 = df[df['Label'] == 'DoS'].drop(['Label'], axis=1)
df4 = df[df['Label'] == 'Fuzzy'].drop(['Label'], axis=1)


def generate_images(df, folder_name, label):
    count = 0
    ims = []
    image_path = f"train/{label}/"
    os.makedirs(image_path, exist_ok=True)

    for i in range(len(df)):
        count += 1
        if count <= 27:
            im = df.iloc[i].values
            ims = np.append(ims, im)
        else:
            ims = np.array(ims).reshape(9, 9, 3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path + str(i) + '.png')
            count = 0
            ims = []


generate_images(df0, "train/0", "Normal")
generate_images(df1, "train/1", "RPM Spoofing")
generate_images(df2, "train/2", "Gear Spoofing")
generate_images(df3, "train/3", "DoS Attack")
generate_images(df4, "train/4", "Fuzzy Attack")


Train_Dir = './train/'
Val_Dir = './test/'
allimgs = []
for subdir in os.listdir(Train_Dir):
    for filename in os.listdir(os.path.join(Train_Dir, subdir)):
        filepath = os.path.join(Train_Dir, subdir, filename)
        allimgs.append(filepath)

print(len(allimgs))  


Numbers = len(allimgs) // 5  

def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print(f"{srcfile} not exist!")
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, dstfile)

val_imgs = random.sample(allimgs, Numbers)
for img in val_imgs:
    dest_path = img.replace(Train_Dir, Val_Dir)
    mymovefile(img, dest_path)
print('Finished creating test set')


def get_224(folder, dstdir):
    imgfilepaths = []
    for root, dirs, imgs in os.walk(folder):
        for thisimg in imgs:
            thisimg_path = os.path.join(root, thisimg)
            imgfilepaths.append(thisimg_path)
    for thisimg_path in imgfilepaths:
        dir_name, filename = os.path.split(thisimg_path)
        dir_name = dir_name.replace(folder, dstdir)
        new_file_path = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        img = cv2.imread(thisimg_path)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(new_file_path, img)
    print('Finished resizing')

DATA_DIR_224 = './train_224/'
get_224(folder='./train/', dstdir=DATA_DIR_224)
DATA_DIR2_224 = './test_224/'
get_224(folder='./test/', dstdir=DATA_DIR2_224)


img1 = Image.open('./train_224/0/27.png')
img2 = Image.open('./train_224/1/83.png')
img3 = Image.open('./train_224/2/27.png')
img4 = Image.open('./train_224/3/27.png')
img5 = Image.open('./train_224/4/27.png')

plt.figure(figsize=(10, 10)) 
plt.subplot(1, 5, 1)
plt.imshow(img1)
plt.title("Normal")
plt.subplot(1, 5, 2)
plt.imshow(img2)
plt.title("RPM Spoofing")
plt.subplot(1, 5, 3)
plt.imshow(img3)
plt.title("Gear Spoofing")
plt.subplot(1, 5, 4)
plt.imshow(img4)
plt.title("DoS Attack")
plt.subplot(1, 5, 5)
plt.imshow(img5)
plt.title("Fuzzy Attack")
plt.show() 