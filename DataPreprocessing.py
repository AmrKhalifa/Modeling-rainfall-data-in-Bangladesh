#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:59:00 2018

@author: assiene
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing

class DataPreprocessing:
    
    def __init__(self):
        pass
    
    def create_dataset_in_time_series_form(self, entire_dataframe, time_horizon=20, time_series_column="Rainfall", output_form="sklearn", preprocess=False):
        start_index = 0
        end_index = time_horizon
        entire_dataframe_rows_number = len(entire_dataframe)
        rising_column_name = "Y_t_plus_1"
        X = []
        Y = pd.DataFrame(columns=[rising_column_name])
        
        while end_index < entire_dataframe_rows_number:
            df_subset = entire_dataframe.iloc[start_index:end_index]
            if preprocess:
                X.append(preprocessing.scale(df_subset.values))
            else:
                X.append(df_subset.values)
            Y = Y.append([{rising_column_name: entire_dataframe[time_series_column].iloc[end_index]}])
            start_index+= 1
            end_index+= 1
            
        X = np.array(X)
            
        if output_form == "sklearn":
            X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
            
        return X, Y