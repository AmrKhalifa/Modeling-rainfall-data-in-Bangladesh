import pandas as pd 
import numpy as np 


def read_data():
    df = pd.read_csv('cleaned_customized_daily_rainfall_data .csv')

    df.replace('?',-99999,inplace = True)
    # columns 0:5 features, column 5 : prediction
    return df


def get_month_data(df):

    df = df.loc [df['StationIndex'] == 1]
    df = df.loc[df['Month'] == 1]

    return df

dataframe = read_data()

monthdata = get_month_data(dataframe)
print(monthdata.head())
print(monthdata.tail())

mean = monthdata["Rainfall"].mean()
std = monthdata["Rainfall"].std()
print(mean)
print(std)





