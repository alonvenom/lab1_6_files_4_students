import pickle
import cv2
import urllib.request
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_anomaly_loop
import pytest
import os
import params_links

@pytest.mark.order(14)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link, max_error",
                         [('test1.mp4',
                           params_links.test1_link(),
                           'anomaly_detection_gt.mp4',
                           params_links.anomaly_detection_gt_link(),
                           1280 * 720 * 5)
                          ])
def test_check_created_motion_video(input_video_name, input_video_link, gt_video_file_name,
                                    gt_video_file_name_link,  max_error):
    current_directory = os.getcwd()
    input_video_path = os.path.join(current_directory, input_video_name)
    if not os.path.exists(input_video_path):
        urllib.request.urlretrieve(input_video_link, input_video_path)
    
    anomaly_video_file = os.path.join(current_directory, 'anomaly_detection.mp4')
    run_main_anomaly_loop(input_video_path)
    
    assert os.path.exists(anomaly_video_file), f'file {anomaly_video_file} does not exist'
    assert os.path.getsize(anomaly_video_file) > 500, f'file {anomaly_video_file} has size of {os.path.getsize(anomaly_video_file)}'
