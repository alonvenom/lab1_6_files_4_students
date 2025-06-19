import pickle
import cv2
import urllib.request
import numpy as np
from main import run_main_anomaly_loop
import pytest
import os
import params_links
import glob


@pytest.mark.order(5)
@pytest.mark.parametrize("input_video_name, input_video_link, jpg_file_name, jpg_gt_file_name, jpg_gt_file_link, size",
                         [('test1.mp4',
                           params_links.test1_link(),
                           '1.0_*.jpg',
                           '1.0_2024-06-12__17_17_46_476904.jpg',
                           params_links.image_link(),
                           150000)
                          ])
def test_check_created_jpg_file(input_video_name, input_video_link, jpg_file_name, jpg_gt_file_name, jpg_gt_file_link, size):
    current_directory = os.getcwd()
    input_video_path = os.path.join(current_directory, input_video_name)
    jpg_gt_file_path = os.path.join(current_directory, jpg_gt_file_name)


    if not os.path.exists(input_video_path):
        urllib.request.urlretrieve(input_video_link, input_video_path)
    if not os.path.exists(jpg_gt_file_path):
        urllib.request.urlretrieve(jpg_gt_file_link, jpg_gt_file_path)

   
    im_gt = cv2.imread(jpg_gt_file_path)
    assert im_gt is not None, f"Ground truth image {jpg_gt_file_path} could not be loaded."

 
    files = glob.glob(os.path.join(current_directory, "bbox_images", jpg_file_name))
    assert files, f"No files matched pattern: {jpg_file_name} in /bbox_images"

  
    found_one_file = False
    for file in files:
        im = cv2.imread(file)
        if im is None or im.shape != im_gt.shape:
            continue
        abs_diff = np.abs(im.astype(np.float32) - im_gt.astype(np.float32))
        if np.sum(abs_diff) < size:
            found_one_file = True
            break

    assert found_one_file, f"No matching file found with difference < {size}"
