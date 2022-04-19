from merging_functions import check_size

test_path = 'yolov5/images_test'
inference_path = 'yolov5/images_inference'
check_size(test_path, mode='test')
check_size(inference_path, mode='inference')