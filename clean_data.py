# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    df = CTG_features
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    c_ctg = df.to_dict('list')
    del c_ctg[extra_feature]
    c_ctg = {k: [elem for elem in v if pd.notnull(elem)] for k, v in c_ctg.items()}
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    # c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    df = CTG_features
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    c_cdf = df.to_dict('list')
    del c_cdf[extra_feature]
    for k , v in c_cdf.items():
        v_clean = [x for x in v if pd.notnull(x)]
        for idx, elem in enumerate(v):
            if np.isnan(elem)==True:
                v[idx] = np.random.choice(v_clean)
            else:
                v[idx]=elem


    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    keys = c_feat.columns
    d_summary = {k: {'min': min(c_feat[k]),'Q1':np.quantile(c_feat[k],0.25),'median':np.quantile(c_feat[k],0.5),'Q3':np.quantile(c_feat[k],0.75),'max': max(c_feat[k])} for k in keys}

    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_no_outlier = c_feat.to_dict('list')
    for k, v in c_no_outlier.items():
        step = 1.5*(d_summary[k]['Q3']-d_summary[k]['Q1'])
        bot_lim = d_summary[k]['Q1'] - step
        upp_lim = d_summary[k]['Q3'] + step
        for idx, elem in enumerate(v):
            if elem<bot_lim or elem>upp_lim:
                v[idx] = np.nan
            else:
                v[idx] = elem

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------

    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
