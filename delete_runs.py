from fileinput import filename
import os 
import shutil

path = 'yolov5/runs/detect'
# path = 'yolov5/images_inference'
to_delete = []

for folder in os.listdir(path):
# for folder in to_delete:
    print(f'Deleting: {folder} ...')
    shutil.rmtree(os.path.join(path, folder))
    print(f'{folder} Deleted.')
        