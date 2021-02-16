"""
Test feature extractor imagenet mobilenet_v2_140_224
"""
import csv
import cv2
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import matplotlib.pyplot as plt

#   args parser
ap = argparse.ArgumentParser()
ap.add_argument("--m", help="1/2/3/4")  # 1:mobilenet_v2_140_224/ 2:inception_v3/ 3:resnet_v2_50/ 4:inception_resnet_v2
m = ap.parse_args().m
URL = ""

#   choose the model we are going to use
if m == "1":  # feature extractor from model mobilenet_v2_140_224
    URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"
elif m == "2":  # feature extractor from model inception_v3
    URL = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3"
elif m == "3":  # feature extractor from model resnet_v2_50
    URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"
elif m == "4":  # feature extractor from model inception_resnet_v2
    URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/3"

#   paths to image directorys , metadata file and spritesheet image
IM_PATH_RGB = '/media/andrea/My Passport/flowers/color/'  # change this to path to the one in you pc where you have
# the original DB
IM_PATH_GRAY = '/media/andrea/My Passport/flowers/gray/'  # change this to path to the one in you pc where you have
# the DB images resized and in grayscale
# IM_PATH_SMALL = '/media/andrea/My Passport/flowers/small/'
SPR_PATH = '/home/andrea/PycharmProjects/cluster_visualization/projector/logdir/spritesheet.png'  # change this to
# path to the one in you pc

#   read metadata where there are the images paths
df = pd.read_csv('metadata.csv', delimiter="\t")

#   get model from imagenet
module = hub.Module(URL)
height, width = hub.get_expected_image_size(module)
ch = 3

filename = tf.placeholder(tf.string)
image_bytes = tf.read_file(filename)
image = tf.image.decode_image(image_bytes, channels=ch)
image = tf.image.resize_bilinear([image], [height, width])
features = module(image)

#   we save in feature_vecs.tsv all the feature vectors we obtain from applying the imagenet model
#   to our images
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
