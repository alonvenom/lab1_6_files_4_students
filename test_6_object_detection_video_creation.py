import pytest
import pickle
import cv2
import urllib.request
import numpy as np
from main import run_main_routine_loop
import os
import params_links


@pytest.mark.order(6)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link, max_error",
                         [('test1.mp4',
                           params_links.test1_link(),
                           'object_detection_gt.mp4',
                           params_links.object_detection_left_gt_link(),
                           1280 * 720 * 5)])
def test_check_created_object_detection_motion_video(input_video_name, input_video_link, gt_video_file_name,
                                                     gt_video_file_name_link, max_error):
    current_directory = os.getcwd()
    input_video_path = os.path.join(current_directory, input_video_name)
    gt_video_path = os.path.join(current_directory, gt_video_file_name)

 
    if not os.path.exists(input_video_path):
        urllib.request.urlretrieve(input_video_link, input_video_path)
    if not os.path.exists(gt_video_path):
        urllib.request.urlretrieve(gt_video_file_name_link, gt_video_path)

   
    motion_video_file = os.path.join(current_directory, 'object_detection.mp4')
    run_main_routine_loop(input_video_path)

    
    assert os.path.exists(motion_video_file), f'File {motion_video_file} does not exist'
    file_size = os.path.getsize(motion_video_file)
    assert file_size > 1024, f'File {motion_video_file} has unexpected small size: {file_size} bytes'


    gt_size = os.path.getsize(gt_video_path)
    diff = abs(file_size - gt_size)
    assert diff < max_error, f'File size difference too big: {diff} > {max_error}'
