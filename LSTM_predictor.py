from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import  LSTM
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


def read_data():
    """ this funcitons reads the dataset file and returns a Pandas data frame the contains the data

    Args:
            NONE
    Returns:
        Pandas dataframe: dataframe the contains the data

    """
    df = pd.read_csv('cleaned_customized_daily_rainfall_data .csv')
    df = df.loc[df['StationIndex'] ==1]
    df.replace('?', -99999, inplace=True)
    # columns 0:5 features, column 5 : prediction
    return df


dataset = read_data()
dataset['Date'] = pd.to_datetime(dataset.Year.astype(str) + '-' + dataset.Month.astype(str))
dataset = dataset[['Date','Rainfall']]
print(dataset.tail())
plt.plot(dataset['Date'],dataset['Rainfall'])
#plt.show()

x= dataset.values

train, test = x[0:5000],x[5000:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()


#x_train.reshape()
######
# Training the model

# model = Sequential()
# model.add(LSTM(
#     input_dim = 4,
#     output_dim = 30,
#     return_sequences = True))
# model.add(Dropout(0.2))
#
# model.add(
#     LSTM(
#         100,
#         return_sequences=False))
# model.add(Dropout(0.2))
#
# model.add(Dense(
#
#     output_dim =1
# ))
# model.add(Activation('linear'))
# model.compile(loss ='mse',optimizer='rmsprop')
#
# model.fit(x_train.reshape(-1,1,4),y_train,batch_size=10000,nb_epoch=10,validation_split=.1)
