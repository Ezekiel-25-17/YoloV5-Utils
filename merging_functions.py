from operator import length_hint
import pandas as pd
import numpy as np
import random
import shutil
import csv
import os


def check_size(path, mode='inference'):
    
    if mode == 'inference':
        for filename in os.listdir(path):
            folderpath = os.path.join(path, filename)
            size = 0
            for element in os.scandir(folderpath):
                size += os.path.getsize(element)
            print(f'{filename}: {size / 1000000000} Gb - {len(os.listdir(folderpath))} Images')
    
    elif mode == 'test':
        for filename in os.listdir(path):
            folderpath = os.path.join(path, filename)
            size = 0
            length = 0
            for year in os.listdir(folderpath):
                folderpath_year = os.path.join(folderpath, year)
                length += len(os.listdir(folderpath_year))
                for element in os.scandir(folderpath_year):
                    size += os.path.getsize(element)
            print(f'{filename}: {size / 1000000000} Gb - {length} Images')



def check_integrity(og_path, new_path, city_img, month, year):
    
    og = len(os.listdir(og_path))
    new = len(os.listdir(new_path))

    if og == new:
        print(f'{city_img} - {month}/{year}: OK')
    else:
        print(f'{city_img} - {month}/{year}: {og-new} Missing')



def check_merge(path, city='', year='', threshold=15):
    num = len(os.listdir(path))
    if num > threshold:
        print(f'{city} - {year}: Successfully Merged')
    else:
        print(f'{city} - {year}: Only {num} files in the folder')
  


def merge_folders(source_path, target_path, same_path=True):

    if same_path:
        folders_to_delete = os.listdir(source_path)

    content_list = {}
    list_dir = os.listdir(source_path)
    for index, val in enumerate(list_dir):
        path = os.path.join(source_path, val)
        content_list[list_dir[index]] = os.listdir(path)

    # loop through the list of folders
    for sub_dir in content_list:
    
        # loop through the contents of the list of folders
        for contents in content_list[sub_dir]:
            path_to_content = sub_dir + "/" + contents  
            dir_to_move = os.path.join(source_path, path_to_content)
    
            # move the file
            shutil.move(dir_to_move, target_path)

    if same_path:
        for folder in folders_to_delete:
            os.rmdir(os.path.join(source_path, folder))



def merge_labels(source_path, target_path, city, year):

    path = os.path.join(source_path, year)
    months = [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]
    df = pd.DataFrame(columns=['photo_filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height'])

    for month in months:
        try:
            tmp = pd.read_csv(os.path.join(source_path, year, month, city + '_' + year + month + '.csv')).drop('Unnamed: 0', axis=1)
        except FileNotFoundError:
            print(f'{city} {month}/{year} Not Found')

        df = pd.concat([df, tmp], ignore_index=True)
    
    df.to_csv(os.path.join(target_path, year, city + '_' + year + '.csv'), index=False)
    print(f'{city} + _ + {year} + .csv: Saved')



def merge_labels_resized(source_path, target_path, city, year=None, mode='months'):

    if mode == 'months':
        path = os.path.join(source_path, city, year)
        months = [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]
        df = pd.DataFrame(columns=['photo_filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height'])

        for month in months:
            try:
                tmp = pd.read_csv(os.path.join(path, month, city + '_' + year + month + '_inferred.csv'))
            except FileNotFoundError:
                print(f'{city} {month}/{year} Not Found')

            df = pd.concat([df, tmp], ignore_index=True)
        
        df.to_csv(os.path.join(target_path, city, year, city + '_' + year + '_inferred.csv'), index=False)
        print(f'{city} + _ + {year} + _inferred.csv: Saved')

    elif mode == 'years':
        path = os.path.join(source_path, city)
        years_ = [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]
        df = pd.DataFrame(columns=['photo_filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height'])

        for year_ in years_:
            try:
                tmp = pd.read_csv(os.path.join(path, year_, city + '_' + year_ + '_inferred.csv'))
            except FileNotFoundError:
                print(f'{city} {year_} Not Found')

            df = pd.concat([df, tmp], ignore_index=True)
        
        df.to_csv(os.path.join(target_path, city + '_inferred.csv'), index=False)
        print(f'{city} + _inferred.csv: Saved')




def update_images(old_path, new_path):
    
    for city in os.listdir(old_path):
        shutil.move(os.path.join(old_path, city), new_path)



def delete_outdated_images(old_path, new_path):

    check = os.listdir(new_path)
    for city in os.listdir(old_path):
        if city in check:
            shutil.rmtree(os.path.join(old_path, city))
        else:
            print('WARNING: Images not updated yet')



def create_test_set(img_source_path, img_target_path, img_cities, csv_source_path, csv_target_path, csv_cities, years, brands, exclude=True):

    if not os.path.exists(img_target_path):
        os.makedirs(img_target_path)
        print(f'Directory {img_target_path} Created')
    else:
        shutil.rmtree(img_target_path)
        os.makedirs(img_target_path)
        print(f'Directory {img_target_path} Replaced')

    df = pd.DataFrame(columns=['photo_filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height'])

    for img_city, csv_city in zip(img_cities, csv_cities):
        for year in years:

            try:
                tmp = pd.read_csv(os.path.join(csv_source_path, year, csv_city + '_' + year + '.csv'))
            except FileNotFoundError:
                continue

            if exclude:
                tmp = tmp[~tmp['class'].isin(brands)].reset_index(drop=True)
            else:
                tmp = tmp[tmp['class'].isin(brands)].reset_index(drop=True)

            imgs = list(tmp['photo_filename'])
            for filename in imgs:
                path = os.path.join(img_source_path, img_city, year, filename)
                shutil.copy(path, img_target_path)

            df = pd.concat([df, tmp], ignore_index=True)
            print(f'{img_city} - {year}: Done')

    df.to_csv(os.path.join(csv_target_path, 'inference_set' + '.csv'), index=False)


### NOT DONE

def add_noise(img_source_path, img_target_path, img_cities, csv_og_path, csv_target_path, csv_cities, years, brands, exclude=True, ratio=0.25):

    itr = int(np.floor((len(os.listdir(img_target_path)) / (len(img_cities) * len(years))) * ratio))
    df = pd.read_csv(os.path.join(csv_target_path, 'inference_set.csv'))
    noise_counter = 0
    for img_city, csv_city in zip(img_cities, csv_cities):
        for year in years:

            try:
                tmp = pd.read_csv(os.path.join(csv_og_path, csv_city + '.csv'))
            except FileNotFoundError:
                continue
            
            if exclude:
                tmp = tmp[tmp['class'].isin(brands)].reset_index(drop=True)
            else:
                tmp = tmp[~tmp['class'].isin(brands)].reset_index(drop=True)
            tmp = list(tmp['photo_filename'])
            try: 
                noise = random.sample(tmp, itr)
            except ValueError:
                noise = tmp
            noise_counter += len(noise)

            for filename in noise:

                d = {'photo_filename' : filename,
                    'class' : 'Noise',
                    'xmin' : np.nan, 
                    'ymin' : np.nan,
                    'xmax' : np.nan,
                    'ymax' : np.nan,
                    'width' : 640,
                    'height' : 640}

                df = pd.concat([df, pd.DataFrame([d])]).reset_index(drop=True)
                path = os.path.join(img_source_path, img_city, year, filename)
                shutil.copy(path, img_target_path)
            
            print(f'{img_city} - {year}: Done')
    
    df.to_csv(os.path.join(csv_target_path, 'inference_set_noised.csv'))
    print(f'Noise Ratio: {(100 * noise_counter) / len(os.listdir(img_target_path))}%')
            

