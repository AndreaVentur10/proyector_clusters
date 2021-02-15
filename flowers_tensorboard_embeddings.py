"""
Adding database of images to the tensorboard projector.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


IM_PATH_RGB = '/media/andrea/My Passport/flowers/color/'
IM_PATH_GRAY = '/media/andrea/My Passport/flowers/gray/'
ME_PATH = '/home/andrea/Desktop/TensorBoard-Embedding-Flowers/metadata.csv'
LOGDIR = r'/home/andrea/Desktop/TensorBoard-Embedding-Flowers/flowers/logdir'
#prepare the images and labels

df = pd.read_csv(ME_PATH, delimiter="\t")
x_test = np.zeros(shape=(733,2048))          #shape is (num images,400*300*3)
y_test = []

# setup the write and embedding tensor

summary_writer = tf.summary.FileWriter(LOGDIR)
# summary_writer: The summary writer used for writing events. config:
embedding_var = tf.Variable(x_test, name='flower_embedding')
#embedding_var = tf.Variable(x_test, name='flower_embedding')  # embedding variable

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

ROWS = 28
COLS = 28

label = ['BO', 'DA', 'GA', 'RO', 'HI', 'HY', 'LI', 'OR', 'PE', 'TU']

SPRITE_H = 28
SPRITE_W = 27
sprite_image = np.ones((28 * SPRITE_H, 28 * SPRITE_W))  # creat blank(temblate) sprite_image

index = 0
labels = []

col_list = ["filename", "category"]

df = pd.read_csv('metadata.csv', usecols=col_list)
files = df['filename']
for i in range(0, SPRITE_H):            # rows
    for j in range(0, SPRITE_W):            # columns
        if i != SPRITE_W:
            im = cv2.imread(IM_PATH_GRAY + files[index], cv2.IMREAD_GRAYSCALE)
            sprite_image[i * ROWS: (i + 1) * ROWS, j * COLS: (j + 1) * COLS] = im

        else:
            if j <= 3:
                im = cv2.imread(IM_PATH_GRAY + files[index], cv2.IMREAD_GRAYSCALE)
                sprite_image[i * ROWS: (i + 1) * ROWS, j * COLS: (j + 1) * COLS] = im
        index += 1


with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(y_test): #labels
        meta.write('{}\t{}\n'.format(index, label))

plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')
plt.show()
