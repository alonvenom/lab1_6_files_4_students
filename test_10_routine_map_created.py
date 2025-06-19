import pickle
import cv2
import urllib.request
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_routine_loop
import pytest
import os
import params_links

@pytest.mark.order(10)
@pytest.mark.parametrize("input_video_name, input_video_link, heatmap_gt_file_name, heatmap_gt_file_link, max_error",
                         [('test1.mp4',
                            params_links.test1_link(),
                           'routine_map_left_road_gt.pkl',
                           params_links.routine_map_left_road_gt_link(),
                           1280 * 720 / 2)])
def test_save_heat_map_as_pkl(input_video_name, input_video_link, heatmap_gt_file_name, heatmap_gt_file_link,
                              max_error):
    current_directory = os.getcwd()
    video_file = os.path.join(current_directory, input_video_name)
    if not os.path.exists(video_file):
       urllib.request.urlretrieve(input_video_link, video_file)
    #run_main_routine_loop(video_file)
    pkl_file_path = os.path.join(os.getcwd(), 'routine_map.pkl')
    assert os.path.exists(pkl_file_path)
    assert os.path.getsize(pkl_file_path) > 1024
