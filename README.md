# YoloV5 Utility Scripts

## Pipeline for Inference & Test with YoloV5

### 1. Preprocessing (*preprocessing_inference/test.py*): 
- Resizing images & labels (640x640)
- Merging the resized images from the directory tree in a single target directory
- Merging the resized labels (if necessary) in a single target directory

### 2. Create Inference/Test Set (*create_inference/test_set.py*):
- Choose brands/images to incluse in the set
- Convert labels from csv to txt format
- Terminal commands to run inference/test

### 3. Postprocessing:
- Convert results from txt back to csv (*convert_labels.py*)
- Custom evaluation functions (*evaluation_functions.py*)

## Function Scripts:
- *preprocessing_functions.py:* functions used during the preprocessing
- *merging_functions.py:* functions used to manage files and folders
- *evaluation_functions.py:* custom functions to evaluate results

## Other Utility Scripts:
- *check_size.py:* compute size of target directories
- *delete_runs.py:* clean up past runs of yolo
- *update_dataset.py:* update an already created inference/test set with new images
