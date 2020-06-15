# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:41:17 2020

@author: Deeps
"""

#train_data for image to number of fish

import os
import numpy as np
import pandas as pd
from skimage import io
import pickle
from matplotlib import pyplot as plt
import PIL
import xml.etree.ElementTree as ET
cwd = os.getcwd()
import xmltodict
import h5py
import keras
import numpy as np
%matplotlib inline

path =  cwd +"\\pascal\\"
image_path = cwd +"\\images\\"

cwd
data = {"image":[],"target":[]}


def parseAnnotation(filename):
    with open(path + filename) as fd:
        doc = xmltodict.parse(fd.read())
    return (doc["annotation"]["filename"],len(doc["annotation"]["object"]))    

def read_image(filename):
    img = io.imread(image_path+'\\images\\'+filename)
    return img
    

for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
       result = parseAnnotation(name)
       data["image"].append(result[0])
       data['target'].append(result[1])

with open( "train_data.pickle" , 'wb') as file:
    pickle.dump(pd.DataFrame(data),file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

#####################################################################
###########################get pickle dump###########################

with open(image_path + "train_data.pickle", 'rb') as file:
    data = pickle.load(file)
    
img = read_image(data['image'][0])

plt.imshow(img)
plt.title("Number of fish: %s"%data['target'][0])
############################ML#######################################
