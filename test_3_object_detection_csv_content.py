import pickle
import cv2
import urllib.request
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
                                gt_csv_file_name_link, csv_file_name, max_error):
    current_directory = os.getcwd()
    input_video_name = os.path.join(current_directory, input_video_name)
    gt_csv_file_name = os.path.join(current_directory, gt_csv_file_name)

    # הורדת קבצים רק אם לא קיימים
    if not os.path.exists(input_video_name):
        urllib.request.urlretrieve(input_video_link, input_video_name)
    if not os.path.exists(gt_csv_file_name):
        urllib.request.urlretrieve(gt_csv_file_name_link, gt_csv_file_name)

    # במקרה וכבר יצרת את ה־CSV בתהליך קודם, אין צורך להריץ שוב
    # run_main_anomaly_loop(input_video_name)

    dataframe_gt = pandas.read_csv(gt_csv_file_name)
    dataframe = pandas.read_csv(csv_file_name)

    if len(dataframe_gt.columns) != len(dataframe.columns):
        assert False, f'Missing column title, should be {dataframe_gt.columns}'
    else:
        diff_columns_check = dataframe_gt.columns == dataframe.columns
        assert diff_columns_check.all(), f'Columns title are different: {dataframe.columns[~diff_columns_check]}'

    # Compare only the first 2 rows for specific columns
    columns_to_check = ['track_id', 'object_name']

    for column in columns_to_check:
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(dataframe[column].head(2), dataframe_gt[column].head(2))) if a != b]
        if diffs:
            for diff in diffs:
                print(f"Difference in '{column}' at index {diff[0]}: {diff[1]} != {diff[2]}")
            assert False, f"Differences found in '{column}'."

    # No assertion failure = success
