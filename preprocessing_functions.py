import os
import glob
import cv2
import shutil
import os.path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from xml.dom.minidom import Document
from tqdm import tqdm
from pandas.core.algorithms import unique


def resize_image_bb(img_path, csv_path, resized_img_path, resized_csv_path, width=640, height=640):
    
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame(columns=['photo_filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

    if not os.path.exists(resized_img_path):
        os.makedirs(resized_img_path)
        print(f'Directory {resized_img_path} Created')
    else:
        shutil.rmtree(resized_img_path)
        os.makedirs(resized_img_path)
        print(f'Directory {resized_img_path} Replaced')

    for filename in os.listdir(img_path): # Add tqdm for progress bar
        
        try:
            img = Image.open(os.path.join(img_path, filename))
        except OSError:
            print(f'{filename} not processed')
            continue

        img_w = img.size[0]
        img_h = img.size[1]

        tmp = df[df['photo_filename']==filename]
        for i, row in tmp.iterrows():
            if row['logo'] is not np.nan:
                d = {'photo_filename' : filename,
                    'class' : row['logo'],
                    'xmin' : np.around((row['xsx'] / img_w) * width), 
                    'ymin' : np.around((row['yup'] / img_h) * height),
                    'xmax' : np.around((row['xrx'] / img_w) * width),
                    'ymax' : np.around((row['ydw'] / img_h) * height),
                    'width' : width,
                    'height' : height}
                new_df = new_df.append(d, ignore_index=True)
        try:
            resized_img = img.resize((width, height))
            resized_img.save(os.path.join(resized_img_path, filename))
        except (OSError, KeyError, ValueError):
            print(f'{filename} not processed')
            continue

    new_df.to_csv(resized_csv_path, index=False)



def resize_image(img_path, resized_img_path, width=640, height=640):

    if not os.path.exists(resized_img_path):
        os.makedirs(resized_img_path)
        print(f'Directory {resized_img_path} Created')
    else:
        shutil.rmtree(resized_img_path)
        os.makedirs(resized_img_path)
        print(f'Directory {resized_img_path} Replaced')

    for filename in os.listdir(img_path):
        
        try:
            img = Image.open(os.path.join(img_path, filename))
        except OSError:
            print(f'{filename} not processed')
            continue

        try:
            resized_img = img.resize((width, height))
            resized_img.save(os.path.join(resized_img_path, filename))
        except (OSError, KeyError, ValueError):
            print(f'{filename} not processed')
            continue



def resize_labels(img_path, csv_path, city, year, month, csv_name, width=640, height=640):

    df = pd.read_csv(os.path.join(csv_path, city + '_inferred_640.csv'))
    new_df = pd.DataFrame(columns=['photo_filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'conf'])

    path = os.path.join(csv_path, city, year, month)
    resized_csv_path = os.path.join(path, csv_name)

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    for filename in os.listdir(img_path):
        
        try:
            img = Image.open(os.path.join(img_path, filename))
        except OSError:
            print(f'{filename} not processed')
            continue

        img_w = img.size[0]
        img_h = img.size[1]

        tmp = df[df['photo_filename']==filename]
        for i, row in tmp.iterrows():
            if row['class'] is not np.nan:
                d = {'photo_filename' : filename,
                    'class' : row['class'],
                    'xmin' : np.around((row['xmin'] / width) * img_w), 
                    'ymin' : np.around((row['ymin'] / height) * img_h),
                    'xmax' : np.around((row['xmax'] / width) * img_w),
                    'ymax' : np.around((row['ymax'] / height) * img_h),
                    'width' : img_w,
                    'height' : img_h,
                    'conf' : row['conf']}
                new_df = new_df.append(d, ignore_index=True)

    new_df.to_csv(resized_csv_path, index=False)



def plot_bb(path, xmin, ymin, xmax, ymax, print_img=True, save_img=False, annotation='', thickness=2):    

    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)

    plt.figure(figsize=(15, 15))

    if print_img:
        print(path)

    image = cv2.imread(path, -1) 

    # Blue color in BGR 
    color1 = (255, 0, 0)

    # Using cv2.rectangle() method 
    # Draw a rectangle with blue line borders of thickness of 2 px 

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color1, thickness) 
    cv2.putText(image, annotation, (xmin, xmax-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1 , 2)

    if print_img:
        plt.imshow(image[...,::-1])

    plt.grid(False)

    if save_img:
        plt.savefig('our_picture.png')



def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh  # h = -(h*dh)
    return (x,y,w,h)



def csv_to_txt(csv_path, out_path, labels):

    # Read Csv 
    df = pd.read_csv(csv_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f'Directory {out_path} Created')
    else:
        shutil.rmtree(out_path)
        os.makedirs(out_path)
        print(f'Directory {out_path} Replaced')

    # Group rows based on filename because single file may have multiple annotations
    for name, group in df.groupby('photo_filename'):
        # Create filename
        fname_out = os.path.join( out_path, name.split(".")[0] + '.txt')
        # Open txt file to write
        with open(fname_out, "w") as f:
            # Iter through each bbox
            for row_index, row in group.iterrows():
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']
                width = row['width']
                height = row['height']
                label = row['class']
                # Get label index
                label_str = str(labels[label])
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                # Convert bbox from pascal voc format to yolo txt format
                bb = convert_coordinates((width,height), b)
                # Write into file
                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')



def txt_to_csv(ann_path, img_path, brands=True, dict={}, conf=True):
    annos = []
    # Read txts  
    for files in os.walk(ann_path):
        for file in files[2]:

            # Read image and get its size attributes
            img_name = os.path.splitext(file)[0] + '.jpg'
            fileimgpath = os.path.join(img_path ,img_name)
            im = Image.open(fileimgpath)
            w = int(im.size[0])
            h = int(im.size[1])

            # Read txt file 
            filelabel = open(os.path.join(ann_path , file), "r")
            lines = filelabel.read().split('\n')
            obj = lines[:len(lines)-1]  
            for i in range(0, int(len(obj))):
                objbud=obj[i].split(' ')
                if brands:                
                    name = dict[str(objbud[0])]
                else:
                    name = str(objbud[0])
                # print(name)
                x1 = float(objbud[1])
                y1 = float(objbud[2])
                w1 = float(objbud[3])
                h1 = float(objbud[4])

                xmin = int((x1*w) - (w1*w)/2.0)
                ymin = int((y1*h) - (h1*h)/2.0)
                xmax = int((x1*w) + (w1*w)/2.0)
                ymax = int((y1*h) + (h1*h)/2.0)

                if conf:
                    confidence = float(objbud[5])
                    annos.append([img_name, w, h, name, xmin, ymin, xmax, ymax, confidence])
                else:
                    annos.append([img_name, w, h, name, xmin, ymin, xmax, ymax])
    if conf:
        column_name = ['photo_filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'conf']
    else:
        column_name = ['photo_filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(annos, columns=column_name)       
    return df


