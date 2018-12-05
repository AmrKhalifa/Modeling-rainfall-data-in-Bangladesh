import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_selection

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

def get_day_data(df,i):
    frame = df.loc[df['Day'] ==i]
    return frame

def get_year_data(df,i):
    frame = df.loc[df['Year'] ==i]
    return frame

def get_month_stats(df):
    months =[]
    means = []

    for i in range(1, 13):
        frame = get_month_data(df, i)
        means.append(frame["Rainfall"].mean())
        months.append(i)

    return np.array([np.array(months),np.array(means)]).T


def get_day_stats(df):
    days = []
    means = []

    for i in range (1,31):
        frame = get_day_data(df,i)
        means.append(frame["Rainfall"].mean())
        days.append(i)

    return np.array([np.array(days),np.array(means)]).T

def get_year_stats(df):
    years = []
    means = []

    for i in range (1999,2016):
        frame = get_year_data(df,i)
        means.append(frame['Rainfall'].mean())
        years.append(i)

    return np.array([np.array(years), np.array(means)]).T

dataframe = read_data()

y = get_year_stats(dataframe)
m = get_month_stats(dataframe)
d = get_day_stats(dataframe)

m1 = feature_selection.mutual_info_regression(y[:,0].reshape(-1,1),y[:,1].reshape(-1,1))
m2 = feature_selection.mutual_info_regression(m[:,0].reshape(-1,1),m[:,1].reshape(-1,1))
m3 = feature_selection.mutual_info_regression(d[:,0].reshape(-1,1),d[:,1].reshape(-1,1))


#print(m1,m2,m3)

plt.plot(d[:,0].reshape(-1,1),d[:,1].reshape(-1,1),linewidth = 2,color = 'r',marker ='x',markersize = 10)
plt.ylim(0,np.max(d[:,1])+1,1)
plt.xlabel('Day',size = 20)
plt.ylabel('Average Rainfall',size =20)
plt.show()
