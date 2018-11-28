import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def get_month_data(df, i):
    """ This function gets the dataframe for a specific month

    Args:
           df : Entire dataset dataframe ,
           i : index of the month of interest (number)
    Returns:
        Pandas dataframe: dataframe the contains the data of the given month

    """
    frame = df.loc[df['Month'] == i]
    return frame


def get_stats(df):
    means = []
    stds = []

    for i in range(1, 13):
        frame = get_month_data(df, i)
        means.append(frame["Rainfall"].mean())
        stds.append(frame["Rainfall"].std())

    return np.array(means), np.array(stds)


def get_feature_vector(df):
    means, stds = get_stats(df)

    #means = means / abs(means).max()
    #stds = stds / abs(stds).max()
    #stds = stds/1.5
    sum_means_stds = (means+stds).reshape(1,-1)
    means_stds = np.concatenate((means, stds)).reshape(1, -1)

    return means_stds

    #return (means + stds).reshape(1,-1)


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


dataframe = read_data()

cluster_input = define_data_frame()

for i in range(1, 36):
    df = dataframe.loc[dataframe['StationIndex'] == i]
    z = get_feature_vector(df)

    station_dataframe = generate_iteration_data_frame(z)
    cluster_input = concat_frames(cluster_input, station_dataframe)

save_dataframe2CSV(cluster_input, "cluster_dataset.csv")

my_frame = pd.read_csv('cluster_dataset.csv')
print(my_frame.values)

for i in range (1,36):
    plt.plot(my_frame.values[i])
plt.show()