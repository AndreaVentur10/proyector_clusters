"""
Prepare images on grayscale and size (28,28)
"""
import os
import csv
import cv2

#   paths to images folder
FL_DB = '/media/andrea/My Passport/flowers/color/'  # change this to path to the one in you pc where
# you have then original DB
FL_SV = '/media/andrea/My Passport/flowers/gray/'
# FL_SM = '/media/andrea/My Passport/flowers/small/'
# FL_MT = '/media/andrea/My Passport/metadata.csv'
#   all flower names of the DB and its respective labels
f = ['bougainvillea', 'daisies', 'gardenias', 'garden_roses', 'hibiscus', 'hydrangeas', 'lilies',
     'orchids', 'peonies', 'tulip']
l = ['BO', 'DA', 'GA', 'RO', 'HI', 'HY', 'LI', 'OR', 'PE', 'TU']

#   list the images
dir_images = os.listdir(FL_DB)
index = 0

#   create metadata.csv file which contains the name and label of each image
with open('metadata.csv', 'w', newline='') as meta:
    writer = csv.writer(meta)
    writer.writerow(["filename", "category"])

    for image in dir_images:
        # print(image)
        im = cv2.imread(FL_DB + image)
        #   if the name contains one of the names of the flowers array then write the label
        for fl in f:
            if fl in image:
                print(image + '  ' + fl)
                writer.writerow([image, l[f.index(fl)]])
                break
        index += 1
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im_gray, (28, 28))
        #   save image
        cv2.imwrite(FL_SV + image, im)
    # print(dir_images)
