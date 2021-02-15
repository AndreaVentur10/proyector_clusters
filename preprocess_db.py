import cv2
import os
import csv
import pandas as pd

FL_DB = '/media/andrea/My Passport/flowers/color/'
FL_SV = '/media/andrea/My Passport/flowers/gray/'
FL_MT = '/media/andrea/My Passport/metadata.csv'
f=['bougainvillea', 'daisies', 'gardenias', 'garden_roses', 'hibiscus', 'hydrangeas', 'lilies', 'orchids', 'peonies', 'tulip' ]
l = [ 'BO', 'DA', 'GA', 'RO', 'HI', 'HY', 'LI', 'OR', 'PE', 'TU']

dir = os.listdir(FL_DB)
index = 0



with open(FL_MT, 'w', newline='') as meta:
    writer = csv.writer(meta)
    writer.writerow(["filename", "category"])
    #meta.write('Index\tLabel\n')

    for image in dir:
        #print(image)
        im= cv2.imread(FL_DB + image)
        for fl in f:
            if fl in image:
                print(image+ '  '+ fl)
                writer.writerow([image, l[f.index(fl)]])
                #meta.write('{}\t{}\n'.format(image, l[f.index(fl)]))
                #meta.write('{}\t{}\n'.format(index, l[f.index(fl)]))
                break
        index += 1
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im_gray, (28, 28))

        cv2.imwrite(FL_SV + image, im)
    #print(dir)
