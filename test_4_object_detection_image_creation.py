import pickle
import cv2
import urllib.request
import numpy as np
from main import run_main_anomaly_loop
import pytest
import os
import params_links
import glob


@pytest.mark.order(4)
@pytest.mark.parametrize("input_video_name, input_video_link, jpg_file_name, size",
                         [('test1.mp4',
                           params_links.test1_link(),
                           '1.0_*.jpg',
                           100)
                          ])
def test_check_created_jpg_file(input_video_name, input_video_link, jpg_file_name, size):
    current_directory = os.getcwd()
    input_video_name = os.path.join(current_directory, input_video_name)

    if not os.path.exists(input_video_name):
        urllib.request.urlretrieve(input_video_link, input_video_name)

    # אין צורך להריץ שוב אם הקבצים כבר נוצרו
    # run_main_anomaly_loop(input_video_name)


    files = glob.glob(f'{current_directory}/bbox_images/{jpg_file_name}')
    assert files, f'{jpg_file_name} does not exist in bbox_images directory'

    if len(files) < size:
        assert False, f'Expected at least {size} files, found {len(files)}'

    pass
