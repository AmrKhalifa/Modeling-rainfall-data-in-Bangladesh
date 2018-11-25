import pandas as pd 
import numpy as np 


def read_data():
    df = pd.read_csv('customized_daily_rainfall_data.csv')
    df.replace('?',-99999,inplace = True)
    # columns 0:5 features, column 5 : prediction 
    x= df[df.columns[0:5]].values
    y= df[df.columns[5:6]].values
    
    return x,y


x,y = read_data()

