import pickle
import cv2
import gdown
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_anomaly_loop
import pytest
import os
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
    input_video_name = os.path.join(current_directory, input_video_name)
    if not os.path.exists(input_video_name):
        gdown.download(input_video_link, input_video_name)
    anomaly_video_file = os.path.join(current_directory, 'anomaly_detection.mp4')
    # run_main_anomaly_loop(input_video_name)
    if os.path.exists(anomaly_video_file):
        gt_video_path = os.path.join(current_directory, gt_video_file_name)  # Replace with the actual path to your video file
        gdown.download(gt_video_file_name_link, gt_video_path)
        cap = cv2.VideoCapture(anomaly_video_file)
        cap_gt = cv2.VideoCapture(gt_video_path)

        # Get the first frame
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading the first frame.")
            exit()
        ret_gt, first_frame_gt = cap_gt.read()
        if not ret_gt:
            print("Error reading the first frame.")
            exit()

        # Get the dimensions of the video frames
        height, width, _ = first_frame.shape
        height_gt, width_gt, _ = first_frame_gt.shape
        q = 0
        failures = []
        failure_values = []
        while True:
            q += 1
            ret, first_frame = cap.read()
            if not ret:
                break
            ret_gt, first_frame_gt = cap_gt.read()
            if not ret_gt:
                break
            diff_image = np.abs(
                (first_frame[:,:,0]>100).astype(np.int64) - (first_frame_gt[:,:,0]>100).astype(np.int64)).astype(np.uint8)
            # diff_image = first_frame_gt[:, width // 2 + 20:,0]>100

            assert np.sum(diff_image) <= 80000 #1280 * 720
    else:
        assert False

    pass