import pickle
import cv2
import gdown
# import matplotlib.pyplot as plt
import numpy as np
from main import run_main_anomaly_loop
from tagging import TaggingSystem
import pytest
import os
import params_links
import pandas

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
        gdown.download(input_video_link, input_video_name)
    if not os.path.exists(gt_csv_file_name):
        gdown.download(gt_csv_file_name_link, gt_csv_file_name)
    anomaly_video_file = os.path.join(current_directory, 'anomaly_detection.mp4')
    # run_main_anomaly_loop(input_video_name) # no need to run this if test_object_detection_csv_creation was run

    #------------------------------un comment this part if you want to run and test---------------------
    TaggingSystem(csv_input_file_name, 'background.png', 'routine_map.pkl')
    cv2.destroyAllWindows()
    #------------------------------un comment this part if you want to run and test---------------------

    dataframe_gt = pandas.read_csv(gt_csv_file_name)
    dataframe = pandas.read_csv(csv_input_file_name)

    # check if number of columns are the same if so check if the title names match
    if len(dataframe_gt.columns) != len(dataframe.columns):
        assert False, f'missing column title should be {dataframe_gt.columns}'
    else:
        diff_columns_check = dataframe_gt.columns == dataframe.columns
        assert (diff_columns_check).all(), f' columns title are different: {dataframe.columns[diff_columns_check==False]}'

    # Define the column you want to check
    column_to_check = 'tag'
    data_frame_list = dataframe[column_to_check].values
    data_frame_gt_list = dataframe_gt[column_to_check].values
    # Assert if the specific column is equal
    # Compare lists element-wise and track indices
    different_objects = [(i, a, b) for i, (a, b) in enumerate(zip(data_frame_list, data_frame_gt_list)) if a != b]

    # assert false if different objects and print their values and indices
    if different_objects:
        for index, obj1, obj2 in different_objects:
            assert False, f"Different objects: Index: {index}, List1: {dataframe.iloc[index]}, List2: {dataframe_gt.iloc[index]}"

    # # assert false if  additional objects and print their indices and values
    # if len(data_frame_list) != len(data_frame_gt_list):
        # additional_indices = list(range(len(data_frame_gt_list), len(data_frame_list))) \
            # if (len(data_frame_list) > len(data_frame_gt_list)) else []
        # additional_list = data_frame_list[len(data_frame_gt_list):] \
            # if (len(data_frame_list) >  len(data_frame_gt_list)) else []
        # for index, obj in zip(additional_indices, additional_list):
            # assert False, (f"Additional objects in {csv_file_name} not in {gt_csv_file_name} Index: {index}, Object: "
                           # f"{dataframe.iloc[index]}")
        # additional_indices_gt = list(range(len(data_frame_list), len(data_frame_gt_list) )) \
            # if (len(data_frame_list) < len(data_frame_gt_list)) else []
        # additional_list_gt = data_frame_gt_list[len(data_frame_list):] \
            # if len(data_frame_list) < len(data_frame_gt_list) else []
        # for index, obj in zip(additional_indices_gt, additional_list_gt):
            # assert False, f"Missing objects from {csv_file_name}: Index: {index}, Object: {dataframe_gt.iloc[index]}"

    pass
