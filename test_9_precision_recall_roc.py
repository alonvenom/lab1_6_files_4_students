import pickle
import cv2
import gdown
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
from precision_recall_roc import generate_precision_recall_auc_graphs
import pytest
import os
import params_links
import glob
import pandas as pd


@pytest.mark.order(9)
@pytest.mark.parametrize("roc_curve_gt_file_name, roc_curve_gt_file_link, roc_curve_file_name, "
                         "precision_recall_gt_file_name, precision_recall_gt_file_link, precision_recall_file_name, "
                         "min_error, tagging_csv_file_name, precision_recall_link_pkl, precision_recall_gt_pkl_file_name",
                         [('roc_curve_gt.png',
                           params_links.roc_link(),
                           'roc_curve.png',
                           'precision_recall_gt.png',
                           params_links.precision_recall_link(),
                           'precision_recall.png',
                           60000,
                           'tagged.csv',
                           params_links.precision_recall_link_pkl(),
                           'fpr_tpr_thresholds.pkl')
                          ])
def test_check_created_roc_precision_curve_file(roc_curve_gt_file_name, roc_curve_gt_file_link, roc_curve_file_name,
                                                precision_recall_gt_file_name, precision_recall_gt_file_link,
                                                precision_recall_file_name, min_error, tagging_csv_file_name,
                                                precision_recall_link_pkl, precision_recall_gt_pkl_file_name):
    # gdown.download(roc_curve_gt_file_link, roc_curve_gt_file_name)
    # gdown.download(precision_recall_gt_file_link, precision_recall_gt_file_name)
    if not os.path.exists(precision_recall_gt_pkl_file_name):
        gdown.download(precision_recall_link_pkl, precision_recall_gt_pkl_file_name)

    y_true = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]

    y_predict = ((100 * np.arange(1, -1 / (len(y_true) - 1), -1 / (len(y_true) - 1))).astype(np.int32)).astype(
        np.float32) / 100

    recall1, precision1, thresholds_pr1, fpr1, tpr1, thresholds_roc1  = (
        generate_precision_recall_auc_graphs(y_true, y_predict, version='v1', show_threshold=False))
    y_true[3] = 1
    recall2, precision2, thresholds_pr2, fpr2, tpr2, thresholds_roc2 = (
        generate_precision_recall_auc_graphs(y_true, y_predict, version='v2', show_threshold=False))

    # Function to compare lists
    def compare_lists(list1, list2):
        return np.allclose(list1, list2, atol=1e-6)

    with open(precision_recall_gt_pkl_file_name, 'rb') as file:
        # Use pickle.dump to serialize and save the data
        [recall1_gld, precision1_gld, thresholds_pr1_gld, fpr1_gld, tpr1_gld, thresholds_roc1_gld,
         recall2_gld, precision2_gld, thresholds_pr2_gld, fpr2_gld, tpr2_gld, thresholds_roc2_gld] =  pickle.load(file)

    #---------------------------------------------------------lab3---------------------------------------------------

    # Create a dictionary to store the results of comparing lists from ground truth data and calculated values
    # Each key represents a metric (e.g., recall1, precision1), and the value stores the result of comparing the lists
    # from ground truth with the calculated values (e.g., recall1_gld with recall1).
    compare_results = {
        'recall1':compare_lists(recall1_gld, recall1),
        'precision1':compare_lists(precision1_gld, precision1),
        'thresholds_pr1':compare_lists(thresholds_pr1_gld, thresholds_pr1),
        'fpr1':compare_lists(fpr1_gld, fpr1),
        'tpr1':compare_lists(tpr1_gld, tpr1),
        'thresholds_roc1':compare_lists(thresholds_roc1_gld, thresholds_roc1),
        'recall2':compare_lists(recall2_gld, recall2),
        'precision2':compare_lists(precision2_gld, precision2),
        'thresholds_pr2':compare_lists(thresholds_pr2_gld, thresholds_pr2),
        'fpr2':compare_lists(fpr2_gld, fpr2),
        'tpr2':compare_lists(tpr2_gld, tpr2),
        'thresholds_roc2':compare_lists(thresholds_roc2_gld, thresholds_roc2)
    }

    # 5. Print the comparison results
    # Iterate through each item in the compare_results dictionary
    # Check if any comparison is False (i.e., lists are different).
    # If no False values exist, that means the lists are identical.
    # Print that the lists are the same if no differences were found.
    # If there is a False (i.e., a difference), raise an assertion error with a message indicating which lists differ.
    for key, value in compare_results.items():
        if value:
            print(f'{key} lists are the same')
        else:
            assert False, f'{key} lists are the diffrence '

    #---------------------------------------------------------lab3---------------------------------------------------
    # assert False # change this
    pass
