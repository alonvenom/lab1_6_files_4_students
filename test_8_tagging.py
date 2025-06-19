import pickle
import cv2
import gdown
import numpy as np
from main import run_main_anomaly_loop
import pytest
import os
import params_links
import pandas as pd
import urllib.request


@pytest.mark.order(8)
@pytest.mark.parametrize("input_video_name, input_video_link, gt_csv_file_name, gt_csv_file_name_link, csv_output_file_name,"
                         " max_error, csv_input_file_name",
                         [('test1.mp4',
                           params_links.test1_link(),
                           'tagged_gt.csv',
                           params_links.csv_anomaly_link(),
                           'tagged.csv',
                           100,
                           "tracked_objects.csv")
                          ])
def test_check_created_csv_tag(input_video_name, input_video_link, gt_csv_file_name,
                               gt_csv_file_name_link,  csv_output_file_name, max_error, csv_input_file_name):
    current_directory = os.getcwd()
    input_video_name = os.path.join(current_directory, input_video_name)

  
    
    if not os.path.exists(input_video_name):
        urllib.request.urlretrieve(input_video_link, input_video_name)

    if not os.path.exists(gt_csv_file_name):
        urllib.request.urlretrieve(gt_csv_file_name_link, gt_csv_file_name)


    run_main_anomaly_loop(input_video_name)


    anomaly_ids = [10.0, 11.0, 27.0, 69.0, 75.0, 90.0, 97.0, 108.0, 116.0]
    df = pd.read_csv(csv_input_file_name)
    df['tag'] = df['track_id'].apply(lambda x: 1 if x in anomaly_ids else 0)
    df.to_csv(csv_input_file_name, index=False)


    dataframe_gt = pd.read_csv(gt_csv_file_name)
    dataframe = pd.read_csv(csv_input_file_name)


    if 'time_date' in dataframe_gt.columns:
        dataframe_gt = dataframe_gt.drop(columns=['time_date'])
    if 'time_date' in dataframe.columns:
        dataframe = dataframe.drop(columns=['time_date'])


    if len(dataframe_gt.columns) != len(dataframe.columns):
        assert False, f'missing column title should be {dataframe_gt.columns}'
    else:
        diff_columns_check = dataframe_gt.columns == dataframe.columns
        assert diff_columns_check.all(), f'columns title are different: {dataframe.columns[diff_columns_check == False]}'

    column_to_check = 'tag'
    data_frame_list = dataframe[column_to_check].values
    data_frame_gt_list = dataframe_gt[column_to_check].values

    different_objects = [(i, a, b) for i, (a, b) in enumerate(zip(data_frame_list, data_frame_gt_list)) if a != b]

    if different_objects:
        for index, obj1, obj2 in different_objects:
            assert False, f"Different objects: Index: {index}, List1: {dataframe.iloc[index]}, List2: {dataframe_gt.iloc[index]}"
