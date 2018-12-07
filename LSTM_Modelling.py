from math import sqrt
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib.pyplot as plt
import pandas as pd

########################

def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df


#########################

# invert differenced value
def add_trend_back(previous_y, y_, shift=1):
	return y_ + previous_y[-shift]

def lstm_fit(train, batch_size, epochs, output):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(10, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	for i in range(10):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model


def full_predict(model, batch_size, row):
	X = row[0:-1]
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def remove_trend(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

def predict(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def re_transform(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

########################################################################################################################

# data loading

def read_data():
    df = pd.read_csv('cleaned_customized_daily_rainfall_data .csv')
    df = df.loc[df['StationIndex'] ==1]
    df.replace('?', -99999, inplace=True)
    # columns 0:5 features, column 5 : prediction
    return df


dataset = read_data()

dataset['Date'] = pd.to_datetime(dataset.Year.astype(str) + '-' + dataset.Month.astype(str))
dataset = dataset['Rainfall']

# transform data to be stationary
ground_truth = dataset.values
diff_values = remove_trend(ground_truth, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:5000], supervised_values[5000:]


# rescale the dataset
scaler, train_scaled, test_scaled = scale(train, test)

lstm_model = lstm_fit(train_scaled, 1, 10, 5)
# predicting the training input to generate information for future forcasting
ecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)


predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    y_ = predict(lstm_model, 1, X)
    # invert scaling
    y_ = re_transform(scaler, X, y_)
    # invert differencing
    y_ = add_trend_back(ground_truth, y_, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(y_)
    expected = ground_truth[len(train) + i + 1]
    print('day=%d, Predicted=%f, Expected=%f' % (i + 1, y_, expected))

# model evaluation
rmse = sqrt(mean_squared_error(ground_truth[5000 + 1:], predictions))
print('Testing RMSE: %.4f' % rmse)

# repeating the experiment
trials = 1
error_values = list()
for r in range(trials):
    # fit the model
    lstm_model = lstm_fit(train_scaled, 1, 10, 5)

    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    predictions = list()
    for i in range(len(test_scaled)):
        # one-step look-ahead
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        y_ = predict(lstm_model, 1, X)
        # re_scaling
        y_ = re_transform(scaler, X, y_)
        # re_trending
        y_ = add_trend_back(ground_truth, y_, len(test_scaled) + 1 - i)

        predictions.append(y_)
    # save model performance
    rmse = sqrt(mean_squared_error(ground_truth[5000 + 1:], predictions))
    print('%d) Testing RMSE: %.3f' % (r + 1, rmse))
    error_values.append(rmse)

plt.plot(ground_truth[5000:], label ='Ground Truth')
plt.plot(predictions,label = 'Prediction')
plt.legend(prop={'size': 20})
plt.xlabel('Days',fontsize=16)
plt.ylabel('Rainfall prediction',fontsize=16)
plt.show()


results = DataFrame()
results['rmse'] = error_values
print(results.describe())
results.boxplot()
plt.legend()
plt.xlabel('RMSE',fontsize = 20)
plt.show()
