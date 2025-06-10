import pickle
import cv2
import gdown
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_routine_loop
import pytest
import os
import params_links
@pytest.mark.order(7)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_video_file_name, gt_video_file_name_link, max_error,"
                         "object_detection_file_name, number_of_error_frames",
                         [('test1.mp4',
                           params_links.test1_link(),
                          'object_detection_both_gt.mp4',
                          params_links.object_detection_left_right_side_gt_link(),
                          1280 * 720 * 8 * 3,
                          'object_detection.mp4',
                           5)
                          ])
def test_check_created_anomaly_detection_motion_video(input_video_name, input_video_link, gt_video_file_name,
                                                      gt_video_file_name_link, max_error, object_detection_file_name,
                                                      number_of_error_frames):
    current_directory = os.getcwd()
    input_video_name = os.path.join(current_directory, input_video_name)
    if not os.path.exists(input_video_name):
        gdown.download(input_video_link, input_video_name)
    anomaly_video_file = os.path.join(current_directory, object_detection_file_name)
    run_main_routine_loop(input_video_name) # leave this commnt on submit
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
        different_frame_counter = 0
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

            # create an absolute difference color image between frame and frame_gt

            frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            frame_gt_gray = cv2.cvtColor(first_frame_gt, cv2.COLOR_BGR2GRAY)
            diff_image = np.sum(np.abs(frame_gray.astype(np.float64) - frame_gt_gray.astype(np.float64)))

            # if sum on absolute difference is less than max_error then add one to different_frame_counter
            is_smaller_than_threshold = diff_image < max_error
            if not is_smaller_than_threshold:
                different_frame_counter+=1
                print(different_frame_counter)

            # if different_frame_counter is larger then
            if different_frame_counter>number_of_error_frames:
                assert False, f'diff_image={diff_image} < max_error {max_error}' # change this

    else:
        assert False, f'{anomaly_video_file} does not exists'

    pass