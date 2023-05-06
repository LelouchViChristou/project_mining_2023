# +
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv')

greece_df = df.loc[df['Entity'] == 'Greece', ['Date', 'Cases', 'Daily tests']].copy()
greece_df = greece_df.reset_index(drop=True)
greece_df = greece_df.drop(greece_df.index[range(0,50)])
greece_df = greece_df.reset_index(drop=True)

greece_df['Positive Ratio'] = (greece_df['Cases'].diff() / greece_df['Daily tests']) * 100
greece_df.iloc[314,3] = float("NaN") #Outlier
greece_df = greece_df.apply(lambda x: x.fillna(method='ffill'))
greece_df = greece_df.apply(lambda x: x.fillna(method='bfill'))
greece_df.drop_duplicates(inplace=True)
#greece_df = greece_df.fillna(0)
train_data = greece_df.iloc[:,3:4].values

# create a MinMax scaler object for all columns
scaler = MinMaxScaler(feature_range=(0,1))

# select all columns to normalize
#columns_to_normalize = ['Daily tests','Cases','Deaths','Positive Ratio','Death Ratio',"Tested ratio"]

# fit and transform all columns with the scaler object
train_data = scaler.fit_transform(train_data)
#print(greece_df.head(20).to_string())

#input_greece_df = greece_df.copy()
#output_greece_df = input_greece_df.pop("Positive Ratio")

pd.options.display.max_columns = 500 #Changes the number of columns diplayed (default is 20)
pd.options.display.max_rows = 500 #Changes the number of rows diplayed (default is 60)
pd.options.display.max_colwidth = 500 #Changes the number of characters in a cell so that the contents don't get truncated 
greece_df

# +
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(6, greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0]):
   x_train.append(train_data[i-6:i,0]) 
   y_train.append(train_data[i+3,0])
    
for i in range(greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0], train_data.size - 3):
   x_test.append(train_data[i-6:i,0]) 
   y_test.append(train_data[i+3,0])

x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

x_test,y_test = np.array(x_test),np.array(y_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# +
from keras.layers import Dense,LSTM,Dropout

# Define the model

model = keras.Sequential(
    [
        layers.LSTM(300, return_sequences=True, input_shape=(x_train.shape[1],1)),
        layers.Dropout(0.2),
        layers.LSTM(300, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(300, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ]
)

model.compile(
    loss = keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate = 0.001),
    metrics="mean_absolute_percentage_error"
)

#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss')

hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_data=(x_test, y_test), verbose=2)

# +
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('LSTM (RNN) Model Loss:')
plt.legend(['Train Set MSE Loss', 'Test Set MSE Loss'], loc='best')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(hist.history['mean_absolute_percentage_error'])
plt.title('LSTM (RNN) Model Train Set Similarity:')
plt.legend(['Train Set Mean Absolute Percentage Error'], loc='best')
plt.ylabel('Cosine Similarity')
plt.xlabel('Epoch')
plt.show()

plt.plot(hist.history['val_mean_absolute_percentage_error'])
plt.title('LSTM (RNN) Model Test Set Similarity:')
plt.legend(['Test Set Mean Test Percentage Error'], loc='best')
plt.ylabel('Cosine Similarity')
plt.xlabel('Epoch')
plt.show()

# +
y_pred = model.predict(x_test)
predicted_price = scaler.inverse_transform(y_pred)
y_test = np.reshape(y_test,(y_test.size,1))
real_price = scaler.inverse_transform(y_test)

plt.plot(real_price, color = 'blue', label = 'Actual Positive Ratio')
plt.plot(predicted_price, color = 'red', label = 'Predicted Positive Ratio')
plt.title('Actual vs Predicted Positive Ratio')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()

plt.plot(scaler.inverse_transform(train_data), color = 'blue', label = 'Actual Positive Ratio')
plt.title('Actual Positive Ratio')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()

# +
y_pred = model.predict(x_train)
predicted_price = scaler.inverse_transform(y_pred)
y_train = np.reshape(y_train,(y_train.size,1))
real_price = scaler.inverse_transform(y_train)

plt.plot(real_price, color = 'blue', label = 'Actual Positive Ratio')
plt.plot(predicted_price, color = 'red', label = 'Predicted Positive Ratio')
plt.title('Actual vs Predicted Positive Ratio On Train Set')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()
