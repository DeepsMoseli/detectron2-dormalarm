# Doortectron2
This work illustrates the use of Detectron2 on a custom dataset. While I worked on this project on data I can not yet share, I felt it important to share my experience with detectron2 so I quickly created a dataset of my own I could share(uncomfortably) so others struggle less with  the detectron library than I did.
 
Detectron2 is a pytorch library form Facebook AI research (FAIR) that primarily re-implements state-of-the-art object object detection algorithms using maskrcnn. While looking at the demo notebook looks fairly straightforward, but Once I tried to use it on a custom labelled dataset, I realised it was harder than I expected. I will go thought the process of coco style annotating a custom dataset and eventually fine tuning a rcnn model for segmentation.  

## Annotations

For annotating training data, I used the [vgg image annotator from oxford](http://www.robots.ox.ac.uk/~vgg/software/via/via.html).  You simply upload your images and start making polyline annotations that per image that eventually export in COCO format in a json file. you do not have to do all the annotating in one go for your dataset, you can save the json file and later upload it together with the images and pickup where you left off.

![via preview](vgg.png)

### Dataset

So as a pet project I decided to build an image based intruder alert system for my dorm room (LOL). I hand labelled about 200 images taken with both my phone and my laptop webcam, of my door both opened and closed. Since I am a grad student, I use a chair to keep my door open, so my categories are ['door', 'opening', 'chair', 'person']. 

Once you are happy with the dataset, you then convert it to polyline coco style os a list of annotations. letsgo.py:

```python

import json
import os
from PIL import Image
import pickle
path = os.getcwd()

# We mostly care about the x and y coordinates of each region

anno_file = "annotations.json"
image_path = "Path_to_images/"

def annotations_file(jsonfile):
    complete_annotations = []
    
    annotations = json.load(open(path+ jsonfile))
    annotations = list(annotations.values())[1]
    
    image_id  = list(range(1,len(annotations)+1))
    
    file_name = [annotations[k]['filename'] for k in annotations]
    
    annotations = [annotations[k] for k in annotations if annotations[k]['regions']]
    annotations = [annotations[k]['regions'] for k in range(len(annotations))]
    
    for k in range(len(annotations)):
        dict_hold = {"file_name":file_name[k],
                     "size": Image.open(image_path + file_name[k]).size,
                     "image_id":image_id[k],
                     "annotations":annotations[k]}
        complete_annotations.append(dict_hold)
    return complete_annotations

annotations = annotations_file(anno_file)

##save annotations as a pickle file
with open( "dorm_annotations.pickle" , 'wb') as file:
    pickle.dump(annotations,file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
```

Eventually I run the model on a video stream that will allow live segmentation. Now lets go straight into it.

__import important, pytorch and detectron libs__

```python
import torch, torchvision
print(torch.__version__,torch.cuda.is_available())
device = torch.device("cuda") 

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import pickle
import random
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
import deepdish as dd
import pycocotools as pycoco
import warnings
import os
warnings.filterwarnings("ignore")

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from skimage import measure
torch.cuda.empty_cache()
```
 
 ### Register your dataset to detectron2
 
 first need to register a dataset in the coco formart to work with predefined data loaders
 
 ```python
 def get_annos():
    with open('dorm_annotations_3.pickle', 'rb') as f:
        annotations = pickle.load(f)
        
    from sklearn.model_selection import train_test_split as tts
    train_annotations,test_annotations = tts(annotations,test_size = 0.3, random_state = 75, shuffle = True)
    data = {'train':train_annotations,'val':test_annotations}
    return data

def category_id(classname):
    if classname =='door':
        return 0
    elif classname =='opening':
        return 1
    elif classname =='person':
        return 2
    else:
        return 3

def create_dataset(annotation):
    dataset_dicts = []
    image_path = os.getcwd().replace("src",'images') + '/'
    for im in range(len(annotation)):
        dict1 = {}
        
        dict1["file_name"] = image_path+annotation[im]["file_name"]
        height, width = cv2.imread(dict1["file_name"]).shape[:2]
        dict1["height"] = height
        dict1["width"] = width
        dict1["image_id"] = (im+1)
        
        objs = []
        for anno in annotation[im]['annotations']:
            #assert not anno["region_attributes"]
            try:
                cat =  category_id(anno['region_attributes']['region'])
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x+0.5, y+0.5) for x,y in zip(px,py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox":[np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode":BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id":cat,
                    "iscrowd":0,
                    }
                objs.append(obj)
            except Exception as e:
                print("image  "+  dict1["file_name"] + ": ", e)
            
        dict1["annotations"] = objs
        
        dataset_dicts.append(dict1)
    return dataset_dicts


def register_dset():
    DatasetCatalog.clear()
    for d_set in data:
        DatasetCatalog.register("doornet_%s"%d_set, lambda d_set=d_set: create_dataset(data[d_set]))
        MetadataCatalog.get("doornet_%s"%d_set).set(thing_classes = ["door",'opening', 'person','chair'])

### load annotations, Register dataset
data = get_annos()
register_dset()
```
 
Now create a detectron config with resnet50 weights for coco instance segmentation and defaultPredictor.

```python
cfg =get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 #model threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
```


 
### Preview
[![watch the video](preview.png)](https://youtu.be/dFU8_TsuHvI)
