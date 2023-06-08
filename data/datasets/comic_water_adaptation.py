import os
import errno

from tqdm import tqdm
import pickle as pkl
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from pymage_size import get_image_size

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

all_class_name =["bicycle", "bird", "car", "cat", "dog", "person"]

def get_annotation(root, image_id, ind):
    annotation_file = os.path.join(root, "Annotations", "%s.xml" % image_id)
    
    et = ET.parse(annotation_file)

    
    objects = et.findall("object")                                              
    
    record = {}
    record["file_name"] = os.path.join(root,  "JPEGImages", "%s.jpg" % image_id)
    img_format = get_image_size(record["file_name"])
    w, h = img_format.get_dimensions()

    record["image_id"] = image_id#ind for pascal evaluation actual image name is needed 
    record["annotations"] = []

    for obj in objects:
        class_name = obj.find('name').text.lower().strip()
        if class_name not in all_class_name:
            print(class_name)
            continue
        if obj.find('pose') is None:
            obj.append(ET.Element('pose'))
            obj.find('pose').text = '0'

        if obj.find('truncated') is None:
            obj.append(ET.Element('truncated'))
            obj.find('truncated').text = '0'

        if obj.find('difficult') is None:
            obj.append(ET.Element('difficult'))
            obj.find('difficult').text = '0'

        bbox = obj.find('bndbox')
        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = max(0,float(bbox.find('xmin').text) - 1) # fixing when -1 in anno
        y1 = max(0,float(bbox.find('ymin').text) - 1) # fixing when -1 in anno
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        box = [x1, y1, x2, y2]
        
        #pascal voc evaluator requires int 
        bbox.find('xmin').text = str(int(x1))
        bbox.find('ymin').text = str(int(y1))
        bbox.find('xmax').text = str(int(x2))
        bbox.find('ymax').text = str(int(y2))


        record_obj = {
        "bbox": box,
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": all_class_name.index(class_name),
        }
        record["annotations"].append(record_obj)

    if len(record["annotations"]):
        #to convert float to int
        et.write(annotation_file)
        record["height"] = h
        record["width"] = w
        return record

    else:
        return None

def files2dict(root,split):

    cache_dir = os.path.join(root, 'cache')

    pkl_filename = os.path.basename(root)+f'_{split}.pkl'
    pkl_path = os.path.join(cache_dir,pkl_filename)

    if os.path.exists(pkl_path):
        with open(pkl_path,'rb') as f:
            return pkl.load(f)
    else:
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(e)
            pass    

    dataset_dicts = []
    image_sets_file = os.path.join( root, "ImageSets", "Main", "%s.txt" % split)

    with open(image_sets_file) as f:
        count = 0

        for line in tqdm(f):
            record = get_annotation(root,line.rstrip(),count)
 
            if record is not None:
                dataset_dicts.append(record)
                count +=1 

    with open(pkl_path, 'wb') as f:
        pkl.dump(dataset_dicts,f)
    return dataset_dicts


def register_dataset(datasets_root):
    dataset_list = ['comic', 
                    'watercolor'
                    ]
    settype = ['train','test']
    
    for name in dataset_list:
        for ind, d in enumerate(settype):
        
                DatasetCatalog.register(name+"_" + d, lambda datasets_root=datasets_root,name=name,d=d \
                    : files2dict(os.path.join(datasets_root,name), d))
                MetadataCatalog.get(name+ "_" + d).set(thing_classes=all_class_name,evaluator_type='pascal_voc')
                MetadataCatalog.get(name+ "_" + d).set(dirname=datasets_root+f'/{name}')
                MetadataCatalog.get(name+ "_" + d).set(split=d)
                MetadataCatalog.get(name+ "_" + d).set(year=2007)