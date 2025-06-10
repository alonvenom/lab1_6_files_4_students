import pytest
import pickle
import cv2
import gdown
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_routine_loop
import pytest
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
    input_video_name = os.path.join(current_directory, input_video_name)
    if not os.path.exists(input_video_name):
        gdown.download(input_video_link, input_video_name)
    motion_video_file = os.path.join(current_directory, 'object_detection.mp4')
    run_main_routine_loop(input_video_name)
    assert os.path.exists(motion_video_file), f'file {motion_video_file} does not exists'
    assert os.path.getsize(motion_video_file) > 1024, f'file {motion_video_file} has size of {os.path.getsize(motion_video_file)}'
