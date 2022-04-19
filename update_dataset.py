from merging_functions import update_images, delete_outdated_images

old_path = 'yolov5/images_old'
new_path = 'yolov5/images'

update_images(old_path, new_path)
delete_outdated_images(old_path, new_path)