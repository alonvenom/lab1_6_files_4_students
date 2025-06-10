import pickle
import cv2
import gdown
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_anomaly_loop
import pytest
import os
import params_links
import pandas

@pytest.mark.order(3)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_csv_file_name, gt_csv_file_name_link,csv_file_name,"
                         " max_error",
                         [('test1.mp4',
                           params_links.test1_link(),
                          'tracked_objects_gt.csv',
                          params_links.csv_anomaly_link(),
                          'tracked_objects.csv',
                           100)
                          ])
def test_check_created_csv_bbox(input_video_name, input_video_link, gt_csv_file_name,
                                    gt_csv_file_name_link,  csv_file_name, max_error):
    current_directory = os.getcwd()
    input_video_name = os.path.join(current_directory, input_video_name)
    gdown.download(input_video_link, input_video_name)
    gdown.download(gt_csv_file_name_link, gt_csv_file_name)
    anomaly_video_file = os.path.join(current_directory, 'anomaly_detection.mp4')
    # run_main_anomaly_loop(input_video_name) # no need to run this if test_object_detection_csv_creation was run

    dataframe_gt = pandas.read_csv(gt_csv_file_name)
    dataframe = pandas.read_csv(csv_file_name)

    if len(dataframe_gt.columns) != len(dataframe.columns):
        assert False, f'missing column title should be {dataframe_gt.columns}'
    else:
        diff_columns_check = dataframe_gt.columns == dataframe.columns
        assert (diff_columns_check).all(), f' columns title are different: {dataframe.columns[diff_columns_check==False]}'

    # Define the columns you want to check
    columns_to_check = ['track_id', 'object_name']

    # Compare only the first 4 rows for these columns
    track_id_diff = [(i, a, b) for i, (a, b) in
                     enumerate(zip(dataframe['track_id'].head(2), dataframe_gt['track_id'].head(2))) if a != b]
    object_name_diff = [(i, a, b) for i, (a, b) in
                        enumerate(zip(dataframe['object_name'].head(2), dataframe_gt['object_name'].head(2))) if a != b]

    # If there are differences, assert False and print the differences
    if track_id_diff or object_name_diff:
        for diff in track_id_diff:
            print(f"Difference in 'track_id' at index {diff[0]}: {diff[1]} != {diff[2]}")
        for diff in object_name_diff:
            print(f"Difference in 'object_name' at index {diff[0]}: {diff[1]} != {diff[2]}")

        # Assert False to indicate that differences were found
        assert False, "Differences found in 'track_id' or 'object_name'."

    pass