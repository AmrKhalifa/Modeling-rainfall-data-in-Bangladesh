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
    
    def create_dataset_without_tai(self, entire_dataframe, past_timeframe=20, stock_price="Close", preprocess=True):
        start_index = 0
        end_index = past_timeframe
        entire_dataframe_rows_number = len(entire_dataframe)
        rising_column_name = "Rising"
        X = []
        Y = pd.DataFrame(columns=[rising_column_name])
        
        while end_index < entire_dataframe_rows_number:
            df_subset = entire_dataframe.iloc[start_index:end_index]
            if preprocess:
                X.append(preprocessing.scale(df_subset.values))
            else:
                X.append(df_subset.values)
            is_price_rising =  1 if (entire_dataframe[stock_price].iloc[end_index] > entire_dataframe[stock_price].iloc[end_index - 1]) else 0
            Y = Y.append([{rising_column_name: is_price_rising}])
            start_index+= 1
            end_index+= 1
            
        return np.array(X),Y