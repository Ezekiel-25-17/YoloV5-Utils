from preprocessing_functions import resize_image_bb, csv_to_txt, plot_bb
from merging_functions import check_integrity, check_merge, merge_folders, merge_labels
import pandas as pd
import numpy as np
from PIL import Image
import os

imgs = '../../cold_data/vision_team/img_instagram'
cities_img = ['IM_miami', 'IM_phoenix']
resized_imgs = 'yolov5/images'

csvs = '../../cold_data/vision_team/img_instagram/GOOGLE_PROCESSED'
cities_csv = ['miami_logo', 'phoenix_logo']
resized_csvs = 'yolov5/labels'


c_done = []
y_done = []
m_done = []

for city_img, city_csv in zip(cities_img, cities_csv):

    img_path = os.path.join(imgs, city_img)
    csv_path = os.path.join(csvs, city_csv + '.csv')

    for year in os.listdir(img_path):

        for month in os.listdir(os.path.join(img_path, year)):

            if (city_img in c_done) and (year in y_done) and (month in m_done):
                continue
            if not os.path.exists(os.path.join(resized_csvs, year)):
                os.makedirs(os.path.join(resized_csvs, year))
            if not os.path.exists(os.path.join(resized_csvs, year, month)):
                os.makedirs(os.path.join(resized_csvs, year, month))

            print(f'Doing: {city_img} {month}/{year}...')
            resize_image_bb(os.path.join(img_path, year, month), csv_path, os.path.join(resized_imgs, city_img, year, month), os.path.join(resized_csvs, year, month, city_csv + '_' + year + month + '.csv'))
            check_integrity(os.path.join(imgs, city_img, year, month), os.path.join(resized_imgs, city_img, year, month), city_img, month, year)
        
        try:
            merge_folders(os.path.join(resized_imgs, city_img, year), os.path.join(resized_imgs, city_img, year))
            print(f'{city_img} - {year} Merged')
        except NotADirectoryError:
            print(f'{city_img} - {year} Already Merged')
        check_merge(os.path.join(resized_imgs, city_img, year), city_img, year)
        
        merge_labels(resized_csvs, resized_csvs, city_csv, year)
