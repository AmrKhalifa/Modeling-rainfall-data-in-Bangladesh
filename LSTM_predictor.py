from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import  LSTM
from keras.models import Sequential
import pandas as pd



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


dataset = read_data()

dataset = dataset[["StationIndex","Year","Month","Day","Rainfall"]]
dataset_np = dataset.values
print(dataset_np.shape)
x_train = dataset_np[:,:4]
print(x_train.shape)
y_train = dataset_np[:,4:]
print(y_train.shape)

#x_train.reshape()
######
# Training the model

model = Sequential()
model.add(LSTM(
    input_dim = 4,
    output_dim = 30,
    return_sequences = True))
model.add(Dropout(0.2))

model.add(
    LSTM(
        100,
        return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(

    output_dim =1
))
model.add(Activation('linear'))
model.compile(loss ='mse',optimizer='rmsprop')

model.fit(x_train.reshape(-1,1,4),y_train,batch_size=10000,nb_epoch=10,validation_split=.1)
