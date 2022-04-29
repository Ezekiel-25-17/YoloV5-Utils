from cgi import test
import os
import shutil

images_test = 'yolov5/images_test'
images_inference = 'yolov5/images_inference'

for city in os.listdir(images_test):

    source_path = os.path.join(images_test, city)
    target_path = os.path.join(images_inference, city)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for year in os.listdir(source_path):
        for img in os.listdir(os.path.join(source_path, year)):
            shutil.copy(os.path.join(source_path, year, img), os.path.join(target_path, img))


### nohup python detect.py --source images_inference/IM_tokyo --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name tokyo --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_shanghai --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name shanghai --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_milan --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name milan --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_london --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name london --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_baltimore --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name baltimore --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/nyc --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name nyc --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_la --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name la --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/sf --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name sf --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/paris --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name paris --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_houston --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name houston --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_miami --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name miami --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_phoenix --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name phoenix --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_charlotte --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name charlotte --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_cleveland --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name cleveland --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_chicago --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name chicago --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_atlanta --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name atlanta --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_austin --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name austin --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_nashville --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name nashville --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_amsterdam --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name amsterdam --save-txt --save-conf --save-crop --nosave &
### nohup python detect.py --source images_inference/IM_sp --weights trained_weights/yoloV5x_best.pt --conf-thres 0.623 --iou-thres 0.60 --name sao_paulo --save-txt --save-conf --save-crop --nosave &


