import pandas as pd
import numpy as np


def read_data():
    """ this funcitons reads the dataset file and returns a Pandas data frame the contains the data

    Args:
            NONE
    Returns:
        Pandas dataframe: dataframe the contains the data

    """
    df = pd.read_csv('cleaned_customized_daily_rainfall_data .csv')
    df.replace('?', -99999, inplace=True)
    # columns 0:5 features, column 5 : prediction
    return df


def get_station_data(df, i):
    """ This function gets the dataframe for a specific month

    Args:
           df : Entire dataset dataframe ,
           i : index of the month of interest (number)
    Returns:
        Pandas dataframe: dataframe the contains the data of the given month

    """
    frame = df.loc[df['StationIndex'] == i]
    return frame


def define_data_frame():
    dataFrame = pd.DataFrame(np.zeros((1, 13)))

    return dataFrame


def generate_iteration_data_frame(feature_vector):
    iteration_dataFrame = pd.DataFrame(feature_vector)
    return iteration_dataFrame


def concat_frames(f1, f2):
    # frames = [f1, f2]
    # frame = pd.concat(frames)
    frame = f1.append(f2)
    return frame


def save_dataframe2CSV(f1, file):
    f1.to_csv(file)


entire_set = read_data()

cluster_1_frame = entire_set.loc[entire_set['StationIndex'] == 1]
cluster_1_list = [1, 2, 3, 4, 5, 6, 10, 11, 14, 15, 16, 17, 18, 19, 23, 26]
for i in cluster_1_list:
    cluster_1_frame = concat_frames(cluster_1_frame, get_station_data(entire_set, i))
save_dataframe2CSV(cluster_1_frame, 'cluster_1.csv')

cluster_2_frame = entire_set.loc[entire_set['StationIndex'] == 7]
save_dataframe2CSV(cluster_2_frame, 'cluster_2.csv')

cluster_3_frame = entire_set.loc[entire_set['StationIndex'] == 9]
cluster_3_list = [12, 13, 20, 21, 22, 28, 30, 31]
for i in cluster_3_list:
    cluster_3_frame = concat_frames(cluster_3_frame, get_station_data(entire_set, i))
save_dataframe2CSV(cluster_3_frame, 'cluster_3.csv')

cluster_4_frame = entire_set.loc[entire_set['StationIndex'] == 24]
cluster_4_list = [27, 33]
for i in cluster_4_list:
    cluster_4_frame = concat_frames(cluster_4_frame, get_station_data(entire_set, i))
save_dataframe2CSV(cluster_4_frame, 'cluster_4.csv')

cluster_5_frame = entire_set.loc[entire_set['StationIndex'] == 25]
cluster_5_list = [29, 30, 34, 35]
for i in cluster_5_list:
    cluster_5_frame = concat_frames(cluster_5_frame, get_station_data(entire_set, i))
save_dataframe2CSV(cluster_5_frame, 'cluster_5.csv')