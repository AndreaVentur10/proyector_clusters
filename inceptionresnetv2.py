"""
Test model imagenet inceptionresnetv2
"""

import os
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.contrib.tensorboard.plugins import projector
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

IM_PATH_RGB = '/media/andrea/My Passport/flowers/color/'
IM_PATH_GRAY = '/media/andrea/My Passport/flowers/gray/'
ME_PATH = '/home/andrea/Desktop/TensorBoard-Embedding-Flowers/metadata.csv'
LOGDIR = r'/home/andrea/Desktop/TensorBoard-Embedding-Flowers/flowers/logdir'
# prepare the images and labels

df = pd.read_csv(ME_PATH, delimiter="\t")
x_test = np.zeros(shape=(733, 1536))  # shape is (num images,400*300*3)
y_test = []
# model = VGG16(weights='imagenet', include_top = 'True')

# INCEPTIONV3------
"""base = InceptionV3(include_top=True, weights='imagenet', )
model = Model(inputs=base.input, outputs=base.get_layer('avg_pool').output)
vecs = []
index = 0

if os.path.getsize('vectors.tsv') == 0:
    with open('vectors.tsv', 'w') as vectors:
        for row in df.iterrows():
            print(index)
            array = row[1].array
            array = array[0].split(',')
            im = image.load_img(IM_PATH_RGB + array[0])
            im = preprocess_input(image.img_to_array(im.resize((299, 299))))
            vec = model.predict(np.expand_dims(im, 0)).squeeze()
            vectors.write('{}\t{}\n'.format(index, vec))
            print(x_test.shape, vec.shape)
            np.append(x_test, [vec], axis=0)
            y_test.append(array[1])
            index += 1
    print(x_test.shape)
    # END------
"""
# ---RESNET 50
model = InceptionResNetV2(include_top=False, pooling='avg' )
    #ResNet50(weights='imagenet', include_top=False, pooling='avg')
# load image setting the image size to 224 x 224
index = 0

if os.path.getsize('vectors3.tsv') == 0:
    with open('vectors3.tsv', 'w') as vectors:
        for row in df.iterrows():
            print(index)
            array = row[1].array
            array = array[0].split(',')
            im = image.load_img(IM_PATH_RGB + array[0], target_size=(224, 224))# convert image to numpy array
            x = image.img_to_array(im)  # the image is now in an array of shape (3, 224, 224)
            # need to expand it to (1, 3, 224, 224) as it's expecting a list
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)  # extract the features
            features = model.predict(x)[0]
            # convert from Numpy to a list of values
            vec = np.char.mod('%f', features)

            vectors.write('{}\t{}\n'.format(index, vec))
            print(x_test.shape, vec.shape)
            np.append(x_test, [vec], axis=0)
            y_test.append(array[1])
            index += 1
    print(x_test.shape)


"""
for row in df.iterrows():
    array = row[1].array
    array = array[0].split(',')
    #im = cv2.imread(IM_PATH_RGB+array[0])
    im = image.load_img(IM_PATH_RGB+array[0], target_size=(224,224))
    im = image.img_to_array(im)
    im= np.expand_dims(im , axis =0)
    #im = cv2.resize(im, (400, 300))
    #im = im / 255
    #im = im.reshape(-1)
    im = preprocess_input(im)
    features = model.predict(im)
    np.append(x_test, features, axis=0)
    print(features.shape)
    y_test.append(array[1])


# setup the write and embedding tensor
"""
summary_writer = tf.summary.FileWriter(LOGDIR)
# summary_writer: The summary writer used for writing events. config:
embedding_var = tf.Variable(x_test, name='flower_embedding')
# embedding_var = tf.Variable(x_test, name='flower_embedding')  # embedding variable

config = projector.ProjectorConfig()  # use the projector
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name  # embedding tensor

embedding.metadata_path = os.path.join(LOGDIR, 'metadata1.tsv')
embedding.sprite.image_path = os.path.join(LOGDIR, 'spritesheet.png')
embedding.sprite.single_image_dim.extend([28, 28])  # size of single image in the sprite

projector.visualize_embeddings(summary_writer, config)  # configuire projector

# run the sesion to create the model check point

with tf.Session() as sesh:  # tensorflow session
    sesh.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sesh, os.path.join(LOGDIR, 'flo_model.ckpt'))

# create the sprite image and the metadata file

rows = 28
cols = 28

label = ['BO', 'DA', 'GA', 'RO', 'HI', 'HY', 'LI', 'OR', 'PE', 'TU']

sprite_h = 28
sprite_w = 27
sprite_image = np.ones((28 * sprite_h, 28 * sprite_w))  # creat blank(temblate) sprite_image

index = 0
labels = []

col_list = ["filename", "category"]

df = pd.read_csv('metadata.csv', usecols=col_list)
files = df['filename']
for i in range(0, sprite_h):  # rows
    for j in range(0, sprite_w):  # columns
        if i != sprite_w:
            im = cv2.imread(IM_PATH_GRAY + files[index], cv2.IMREAD_GRAYSCALE)
            sprite_image[i * rows: (i + 1) * rows, j * cols: (j + 1) * cols] = im

        else:
            if j <= 3:
                im = cv2.imread(IM_PATH_GRAY + files[index], cv2.IMREAD_GRAYSCALE)
                sprite_image[i * rows: (i + 1) * rows, j * cols: (j + 1) * cols] = im
        index += 1

with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(y_test):  # labels
        meta.write('{}\t{}\n'.format(index, label))

plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')
plt.show()
