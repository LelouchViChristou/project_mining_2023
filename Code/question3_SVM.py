# +
import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

df = pd.read_csv('data.csv')

greece_df = df.loc[df['Entity'] == 'Greece', ['Date', 'Cases', 'Daily tests']].copy()
greece_df = greece_df.reset_index(drop=True)
greece_df = greece_df.drop(greece_df.index[range(0,60)])
greece_df = greece_df.reset_index(drop=True)

greece_df['Positive Ratio'] = (greece_df['Cases'].diff() / greece_df['Daily tests']) * 100
greece_df.iloc[304,3] = float("NaN") #Outlier
#greece_df = greece_df.fillna(0)

# select all columns to normalize
#columns_to_normalize = ['Daily tests','Cases','Deaths','Positive Ratio','Death Ratio',"Tested ratio"]

# fit and transform all columns with the scaler object
#print(greece_df.head(20).to_string())

#input_greece_df = greece_df.copy()
#output_greece_df = input_greece_df.pop("Positive Ratio")

#pd.options.display.max_columns = 500 #Changes the number of columns diplayed (default is 20)
#pd.options.display.max_rows = 500 #Changes the number of rows diplayed (default is 60)
#pd.options.display.max_colwidth = 500 #Changes the number of characters in a cell so that the contents don't get truncated 
model_data = greece_df.iloc[:,3:4]
model_data = model_data.apply(lambda x: x.fillna(method='ffill'))
model_data = model_data.apply(lambda x: x.fillna(method='bfill'))
model_data.drop_duplicates(inplace=True)

# create a MinMax scaler object for all columns
scaler = MinMaxScaler(feature_range=(0,1))

# select all columns to normalize

# fit and transform all columns with the scaler object
model_data = scaler.fit_transform(model_data)

# +
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(6, greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0]):
   x_train.append(model_data[i-6:i]) 
   y_train.append(model_data[i+3])
    
for i in range(greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0], model_data.size - 3):
   x_test.append(model_data[i-6:i]) 
   y_test.append(model_data[i+3])

x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],-1))

x_test,y_test = np.array(x_test),np.array(y_test)
x_test = np.reshape(x_test,(x_test.shape[0],-1))

# +
# Define the model

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train.ravel())

# +
from statsmodels.tools.eval_measures import mse
y_pred = regressor.predict(x_test)
y_pred = np.reshape(y_pred,(y_pred.size,1))
y_pred = scaler.inverse_transform(y_pred)

y_test = scaler.inverse_transform(y_test)

print(mse(y_pred,y_test))

# +
import matplotlib.pyplot as plt

plt.plot(y_test, color = 'blue', label = 'Actual Positive Ratio')
plt.plot(y_pred, color = 'red', label = 'Predicted Positive Ratio')
plt.title('SVR (RBF) Model Actual vs Predicted Comparisson on Test Set')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()

# +
y_pred = regressor.predict(np.concatenate((x_train,x_test)))
y_pred = np.reshape(y_pred,(y_pred.size,1))
y_pred = scaler.inverse_transform(y_pred)
y_plot = np.reshape(model_data,(model_data.size,1))
y_plot = scaler.inverse_transform(y_plot)
y_plot = y_plot[6:]
y_plot = y_plot[:y_plot.size-3]

plt.plot(y_plot, color = 'blue', label = 'Actual Positive Ratio')
plt.plot(y_pred, color = 'red', label = 'Predicted Positive Ratio')
plt.title('SVR (RBF) Model Actual vs Predicted Positive Ratio On All Data')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()
