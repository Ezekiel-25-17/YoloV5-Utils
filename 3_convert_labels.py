from unicodedata import name
from preprocessing_functions import txt_to_csv, resize_labels
from merging_functions import merge_labels_resized
import pandas as pd
import os

cities = {'sf':'sf', 'nyc':'nyc', 'tokyo':'IM_tokyo', 'shanghai':'IM_shanghai', 'milan':'IM_milan', 'london':'IM_london', 'la':'IM_la', 'baltimore':'IM_baltimore', 'paris':'paris', 'chicago':'IM_chicago', 'charlotte':'IM_charlotte', 
          'cleveland':'IM_cleveland', 'miami':'IM_miami', 'phoenix':'IM_phoenix', 'houston':'IM_houston', 'atlanta':'IM_atlanta', 'austin':'IM_austin', 'nashville':'IM_nashville', 'amsterdam':'IM_amsterdam', 'sao_paulo':'IM_sp'}

brands = {'0':'Adidas', '1':'Apple Inc-', '2':'Coca-Cola', '3':'Emirates', '4':'Hard Rock Cafe', '5':'Mercedes-Benz', '6':'NFL', 
          '7':'Nike', '8':'Pepsi', '9':'Puma', '10':'Starbucks', '11':'The North Face', '12':'Toyota', '13':'Under Armour'}

c_todo = []

# batch = ['tokyo', 'shanghai']

path = 'yolov5/runs/detect'
imgs = 'yolov5/images_inference'
csvs = 'yolov5/labels_inference'  
imgs_1 = '../../cold_data/vision_team/img_instagram'
imgs_2 = '../../cold_data/vision_team/img_instagram/IMG_JPG'


for filename in c_todo:
    ann_path = os.path.join(path, filename, 'labels')
    city = cities[filename]
    img_path = os.path.join(imgs, city)
    df = txt_to_csv(ann_path, img_path, brands=True, dict=brands, conf=True)
    df.to_csv(os.path.join(csvs, filename + '_inferred_640.csv'), index=False)
    print(f'{filename}: Converted to Csv')


for city in c_todo:

    # if city in batch:
        # img_path = os.path.join(imgs_1, cities[city])
    # else:
        # img_path = os.path.join(imgs_2, cities[city])
        
    img_path = os.path.join(imgs_1, cities[city])

    for year in os.listdir(img_path):
        for month in os.listdir(os.path.join(img_path, year)):
            csv_name = city + '_' + year + month + '_inferred.csv'
            resize_labels(os.path.join(img_path, year, month), csvs, city, year, month, csv_name)

        merge_labels_resized(source_path=csvs, target_path=csvs, city=city, year=year, mode='months')
        print(f'{city} - {year}: Done')

    merge_labels_resized(source_path=csvs, target_path=csvs, city=city, year=None, mode='years')
    print(f'{city}: Done')
    
        