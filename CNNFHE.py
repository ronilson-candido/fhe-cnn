import numpy as np
import pandas as pd
import os
import cv2
import math
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
import seal
from seal import EncryptionParameters, SEALContext, KeyGenerator, Encryptor, Decryptor, Evaluator, CKKSEncoder
from seal import scheme_type, CoeffModulus

warnings.filterwarnings("ignore")

# Setup CKKS encryption
def ckks_setup():
    # Define encryption parameters
    parms = EncryptionParameters(scheme_type.ckks)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 60]))

    # Create SEAL context
    context = SEALContext(parms)
    
    # Key generation
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.relin_keys()

    # Create encoder, encryptor, decryptor, and evaluator
    encoder = CKKSEncoder(context)
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)

    return encoder, encryptor, decryptor, evaluator, context

# Function to encrypt data using CKKS
def encrypt_data(data, encoder, encryptor, scale=2**40):
    plain = seal.Plaintext()
    encrypted_data = []
    for value in data:
        encoder.encode(value, scale, plain)
        encrypted = seal.Ciphertext()
        encryptor.encrypt(plain, encrypted)
        encrypted_data.append(encrypted)
    return encrypted_data

# Function to decrypt data
def decrypt_data(encrypted_data, decoder, decryptor):
    decoded_data = []
    for encrypted in encrypted_data:
        plain = seal.Plaintext()
        decryptor.decrypt(encrypted, plain)
        decoded_value = []
        decoder.decode(plain, decoded_value)
        decoded_data.append(decoded_value)
    return np.array(decoded_data)

# Function to add encrypted columns (homomorphic operation)
def add_encrypted_columns(enc_data1, enc_data2, evaluator):
    encrypted_sum = []
    for i in range(len(enc_data1)):
        sum_result = seal.Ciphertext()
        evaluator.add(enc_data1[i], enc_data2[i], sum_result)
        encrypted_sum.append(sum_result)
    return encrypted_sum

# Initialize CKKS setup
encoder, encryptor, decryptor, evaluator, context = ckks_setup()

# Read dataset
df = pd.read_csv('data/Car_Hacking_5%.csv')
print(df.Label.value_counts())  # Labels of dataset

# Transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encrypt the scaled data using CKKS
df_encrypted = df[numeric_features].apply(lambda x: encrypt_data(x, encoder, encryptor))

# Process dataset by attack type
df0 = df[df['Label'] == 'R'].drop(['Label'], axis=1)
df1 = df[df['Label'] == 'RPM'].drop(['Label'], axis=1)
df2 = df[df['Label'] == 'gear'].drop(['Label'], axis=1)
df3 = df[df['Label'] == 'DoS'].drop(['Label'], axis=1)
df4 = df[df['Label'] == 'Fuzzy'].drop(['Label'], axis=1)

# Create function to generate 9x9 images for each class
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

# Generate images for each attack type
generate_images(df0, "train/0", "Normal")
generate_images(df1, "train/1", "RPM Spoofing")
generate_images(df2, "train/2", "Gear Spoofing")
generate_images(df3, "train/3", "DoS Attack")
generate_images(df4, "train/4", "Fuzzy Attack")

# Create folders to store images
Train_Dir = './train/'
Val_Dir = './test/'
allimgs = []
for subdir in os.listdir(Train_Dir):
    for filename in os.listdir(os.path.join(Train_Dir, subdir)):
        filepath = os.path.join(Train_Dir, subdir, filename)
        allimgs.append(filepath)

print(len(allimgs))  # Total number of images

# Split a test set from the dataset, train/test size = 80%/20%
Numbers = len(allimgs) // 5  # Test set size (20%)

# Function to move files to test set
def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print(f"{srcfile} not exist!")
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, dstfile)

# Create the test set
val_imgs = random.sample(allimgs, Numbers)
for img in val_imgs:
    dest_path = img.replace(Train_Dir, Val_Dir)
    mymovefile(img, dest_path)
print('Finished creating test set')

# Resize images to 224x224 for CNN training
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

# Read images for each category and visualize
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
plt.show()  # Display images
