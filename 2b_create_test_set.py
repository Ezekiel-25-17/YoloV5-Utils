import os
from merging_functions import create_test_set
from preprocessing_functions import csv_to_txt

img_source_path = 'yolov5/images'
img_target_path = 'yolov5/data/images'
img_cities = ['IM_charlotte', 'IM_cleveland', 'IM_houston', 'IM_miami', 'IM_chicago', 'IM_phoenix']

csv_og_path = '../../cold_data/vision_team/img_instagram/GOOGLE_PROCESSED'
csv_source_path = 'yolov5/labels'
csv_target_path = 'yolov5/labels'
txt_target_path = 'yolov5/data/labels'
csv_cities = ['charlotte_logo', 'cleveland_logo', 'houston_logo', 'miami_logo', 'chicago_logo', 'phoenix_logo']

years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']

brands = ['Adidas', 'Apple Inc.', 'Coca-Cola', 'Emirates', 'Hard Rock Cafe', 'Mercedes-Benz', 'NFL', 
          'Nike', 'Pepsi', 'Puma', 'Starbucks', 'The North Face', 'Toyota', 'Under Armour']

### YoloV5L
# labels = {'Adidas' : 0, 'Apple Inc-' : 1, 'Coca-Cola' : 2, 'Emirates' : 3, 'Hard Rock Cafe' : 4, 'Mercedes-Benz' : 5, 'NFL' : 6, 
          # 'Nike' : 7, 'Pepsi' : 8, 'Puma' : 9, 'Starbucks' : 10, 'The North Face' : 11, 'Toyota' : 12, 'Under Armour': 13}

### nohup python val.py --weights trained_weights/yoloV5l_best.pt --task test --data test.yaml --verbose --save-txt --save-conf &

### YoloV5X
labels = {'Adidas' : 0, 'Puma' : 1, 'Starbucks' : 2, 'The North Face' : 3, 'Toyota' : 4, 'Under Armour': 5, 'Apple Inc.' : 6, 
          'Coca-Cola' : 7, 'Emirates' : 8, 'Hard Rock Cafe' : 9, 'Mercedes-Benz' : 10, 'NFL' : 11, 'Nike' : 12, 'Pepsi' : 13}

### nohup python val.py --weights trained_weights/yoloV5x_best.pt --task test --data test.yaml --verbose --save-txt --save-conf &

create_test_set(img_source_path, img_target_path, img_cities, csv_source_path, csv_target_path, csv_cities, years, brands, exclude=False)
csv_to_txt(os.path.join(csv_target_path, 'inference_set.csv'), txt_target_path, labels)
