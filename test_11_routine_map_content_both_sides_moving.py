import pickle
import cv2
import urllib.request
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_routine_loop
import pytest
import os
import params_links

@pytest.mark.order(11)
@pytest.mark.parametrize("input_video_name, input_video_link, heatmap_gt_file_name, heatmap_gt_file_link, max_error",
                         [('test1.mp4',
                           params_links.test1_link(),
                           'routine_map_left_right_roads_gt.pkl',
                           params_links.routine_map_left_right_roads_gt_link(),
                           1280 * 720 / 2)])
def test_run_main_anomaly_loop_saves_good_heat_map_as_pkl(input_video_name, input_video_link,
                                                          heatmap_gt_file_name, heatmap_gt_file_link,
                                                          max_error):
    current_directory = os.getcwd()
    video_file = os.path.join(current_directory, input_video_name)
    if not os.path.exists(input_video_name):
        urllib.request.urlretrieve(input_video_link, input_video_name)

   
    # run_main_routine_loop(video_file)

    pkl_file_path = os.path.join(os.getcwd(), 'routine_map.pkl')
    assert os.path.exists(pkl_file_path)
    if os.path.exists(pkl_file_path):
        with open('routine_map.pkl', 'rb') as f:
            heatmap = pickle.load(f)
        if not os.path.exists(heatmap_gt_file_name):
            urllib.request.urlretrieve(heatmap_gt_file_link, heatmap_gt_file_name)

        with open(heatmap_gt_file_name, 'rb') as f:
            heatmap_gt = pickle.load(f)

        height, width = heatmap_gt.shape
        right_half_heatmap = heatmap[:, width // 2 + 20:]
        right_half_heatmap_gt = heatmap_gt[:, width // 2 + 20:]
        assert np.sum(np.abs(right_half_heatmap - right_half_heatmap_gt)) <= max_error, 'heatmap is not equal'
    else:
        assert False, 'routine_map.pkl does not exist'
