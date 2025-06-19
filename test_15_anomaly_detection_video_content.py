import pickle
import cv2
import urllib.request
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_anomaly_loop
import pytest
import os
import sys
import params_links

@pytest.mark.order(15)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link, max_error",
                         [('test1.mp4',
                           params_links.test1_link(),
                           'anomaly_detection_gt.mp4',
                           params_links.anomaly_detection_gt_link(),
                           1280 * 720 * 5)
                          ])
def test_check_created_anomaly_detection_motion_video(input_video_name, input_video_link, gt_video_file_name,
                                                      gt_video_file_name_link, max_error):
    current_directory = os.getcwd()
    input_video_path = os.path.join(current_directory, input_video_name)
    if not os.path.exists(input_video_path):
        urllib.request.urlretrieve(input_video_link, input_video_path)
    
    anomaly_video_file = os.path.join(current_directory, 'anomaly_detection.mp4')
    # run_main_anomaly_loop(input_video_path)
    
    if os.path.exists(anomaly_video_file):
        gt_video_path = os.path.join(current_directory, gt_video_file_name)
        if not os.path.exists(gt_video_path):
            urllib.request.urlretrieve(gt_video_file_name_link, gt_video_path)
        
        cap = cv2.VideoCapture(anomaly_video_file)
        cap_gt = cv2.VideoCapture(gt_video_path)

        ret, first_frame = cap.read()
        if not ret:
            print("Error reading the first frame of anomaly video.")
            sys.exit(1)
        ret_gt, first_frame_gt = cap_gt.read()
        if not ret_gt:
            print("Error reading the first frame of ground truth video.")
            sys.exit(1)

        height, width, _ = first_frame.shape
        height_gt, width_gt, _ = first_frame_gt.shape

        frame_count = 0
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break
            ret_gt, frame_gt = cap_gt.read()
            if not ret_gt:
                break
            
            diff_image = np.abs(
                (frame[:, :, 0] > 100).astype(np.int64) - (frame_gt[:, :, 0] > 100).astype(np.int64)
            ).astype(np.uint8)
            
            assert np.sum(diff_image) <= 80000, f"Frame {frame_count} differs too much between anomaly and ground truth videos."

    else:
        assert False, f"File {anomaly_video_file} does not exist."
