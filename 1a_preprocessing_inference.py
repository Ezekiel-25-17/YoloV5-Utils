from preprocessing_functions import resize_image
from merging_functions import check_integrity, check_merge, merge_folders
import pandas as pd
import numpy as np
from PIL import Image
import os

imgs = '../../cold_data/vision_team/img_instagram'
# imgs = '../../cold_data/vision_team/img_instagram/IMG_jpg'
cities_img = []
resized_imgs = 'yolov5/images_inference'
years = []
final_merge = True

c_done = []
y_done = []

for city_img in cities_img:

    img_path = os.path.join(imgs, city_img)

    for year in os.listdir(img_path):
    # for year in years:

        for month in os.listdir(os.path.join(img_path, year)):

            if (city_img in c_done) and (year in y_done):
                continue

            print(f'Doing: {city_img} {month}/{year}...')
            resize_image(os.path.join(img_path, year, month), os.path.join(resized_imgs, city_img, year, month))
            check_integrity(os.path.join(imgs, city_img, year, month), os.path.join(resized_imgs, city_img, year, month), city_img, month, year)
        
        try:
            merge_folders(os.path.join(resized_imgs, city_img, year), os.path.join(resized_imgs, city_img, year))
            print(f'{city_img} - {year} Merged')
        except NotADirectoryError:
            print(f'{city_img} - {year} Already Merged')
        check_merge(os.path.join(resized_imgs, city_img, year), city=city_img, year=year)

    if final_merge:
        try:
            merge_folders(os.path.join(resized_imgs, city_img), os.path.join(resized_imgs, city_img))
            print(f'{city_img} Merged')
        except NotADirectoryError:
            print(f'{city_img} Already Merged')
        check_merge(os.path.join(resized_imgs, city_img), city=city_img)