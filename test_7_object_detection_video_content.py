import pickle
import cv2
import urllib.request
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
    input_video_path = os.path.join(current_directory, input_video_name)
    gt_video_path = os.path.join(current_directory, gt_video_file_name)
    anomaly_video_file = os.path.join(current_directory, object_detection_file_name)

    if not os.path.exists(input_video_path):
        urllib.request.urlretrieve(input_video_link, input_video_path)
    if not os.path.exists(gt_video_path):
        urllib.request.urlretrieve(gt_video_file_name_link, gt_video_path)

    run_main_routine_loop(input_video_path)

    assert os.path.exists(anomaly_video_file), f'{anomaly_video_file} does not exist'

    cap = cv2.VideoCapture(anomaly_video_file)
    cap_gt = cv2.VideoCapture(gt_video_path)

    
    different_frame_counter = 0
    frame_index = 0

    while True:
        ret1, frame = cap.read()
        ret2, frame_gt = cap_gt.read()

        if not ret1 or not ret2:
            break  

        if frame.shape != frame_gt.shape:
            assert False, f"Frame shape mismatch at frame {frame_index}: {frame.shape} != {frame_gt.shape}"

       
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gt_gray = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)

       
        diff = np.abs(frame_gray.astype(np.float64) - frame_gt_gray.astype(np.float64))
        diff_sum = np.sum(diff)

        if diff_sum > max_error:
            different_frame_counter += 1
            print(f"Frame {frame_index}: diff_sum={diff_sum}")

        frame_index += 1

    assert different_frame_counter <= number_of_error_frames, (
        f"Too many different frames: {different_frame_counter} > {number_of_error_frames}"
    )
