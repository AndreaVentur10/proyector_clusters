"""
Test feature extractor imagenet mobilenet_v2_140_224
"""
import csv
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import matplotlib.pyplot as plt

#   Paths to image directorys , metadata file and spritesheet image
IM_PATH_RGB = '/media/andrea/My Passport/flowers/color/'
IM_PATH_GRAY = '/media/andrea/My Passport/flowers/gray/'
IM_PATH_SMALL = '/media/andrea/My Passport/flowers/small/'
ME_PATH = '/home/andrea/Desktop/TensorBoard-Embedding-Flowers/metadata.csv'
SPR_PATH = '/home/andrea/PycharmProjects/cluster_visualization/projector/logdir/spritesheet.png'

#   read metadata where there are the images paths
df = pd.read_csv(ME_PATH, delimiter="\t")

#   get model from imagenet
module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2")
height, width = hub.get_expected_image_size(module)
ch = 3

filename = tf.placeholder(tf.string)
image_bytes = tf.read_file(filename)
image = tf.image.decode_image(image_bytes, channels=ch)
image = tf.image.resize_bilinear([image], [height, width])
features = module(image)

#   we save in feature_vecs.tsv all the feature vectors we obtain from applying the imagenet model to our images
with open('feature_vecs.tsv', 'w') as fw:
    csv_writer = csv.writer(fw, delimiter='\t')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for row in df.iterrows():
            array = row[1].array
            array = array[0].split(',')
            fvecs = sess.run(features, feed_dict={filename: IM_PATH_RGB + array[0]})
            csv_writer.writerows(fvecs)

# create the sprite image
ROWS = 28
COLS = 28

label = ['BO', 'DA', 'GA', 'RO', 'HI', 'HY', 'LI', 'OR', 'PE', 'TU']

SPRITE_H = 28
SPRITE_W = 27
sprite_image = np.ones((28 * SPRITE_H, 28 * SPRITE_W))  # creat blank(temblate) sprite_image

index = 0
labels = []
#   read metadata again for adding each image of the database to the spritesheet
col_list = ["filename", "category"]
df = pd.read_csv('metadata.csv', usecols=col_list)
files = df['filename']

for i in range(0, SPRITE_H):  # rows
    for j in range(0, SPRITE_W):  # columns
        if i != SPRITE_W:
            im = cv2.imread(IM_PATH_GRAY + files[index], cv2.IMREAD_GRAYSCALE)
            # im = cv2.imread(IM_PATH_SMALL + files[index])
            sprite_image[i * ROWS: (i + 1) * ROWS, j * COLS: (j + 1) * COLS] = im

        else:
            if j <= 3:
                im = cv2.imread(IM_PATH_GRAY + files[index], cv2.IMREAD_GRAYSCALE)
                # im = cv2.imread(IM_PATH_SMALL + files[index])
                sprite_image[i * ROWS: (i + 1) * ROWS, j * COLS: (j + 1) * COLS] = im
        index += 1

#   save the sprite_image and show it on gray scale
cv2.imwrite(SPR_PATH, sprite_image)
plt.imshow(sprite_image, cmap='gray')
plt.show()
