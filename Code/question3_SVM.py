# +
import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Read the CSV file into a pandas dataframe
df = pd.read_csv('data.csv')

# Create a pandas dataframe with the data I need for Greece only
greece_df = df.loc[df['Entity'] == 'Greece', ['Date', 'Cases', 'Daily tests']].copy()
greece_df = greece_df.reset_index(drop=True)

# Remove initial outlier values (tested better performance this way)
greece_df = greece_df.drop(greece_df.index[range(0,60)])
greece_df = greece_df.reset_index(drop=True)

# Compute positive ratio
greece_df['Positive Ratio'] = (greece_df['Cases'].diff() / greece_df['Daily tests']) * 100

# Remove another outlier
greece_df.iloc[304,3] = float("NaN") 

# Keep only positive ratio
model_data = greece_df.iloc[:,3:4]

# Forward fill and backward fill to remove NaN values and drop duplicates
model_data = model_data.apply(lambda x: x.fillna(method='ffill'))
model_data = model_data.apply(lambda x: x.fillna(method='bfill'))

# MinMax scale data to range [0,1]
scaler = MinMaxScaler(feature_range=(0,1))
model_data = scaler.fit_transform(model_data)

# +
#Create input and output arrays for train and test sets
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

#Reshape arrays into correct shapes
x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],-1))

x_test,y_test = np.array(x_test),np.array(y_test)
x_test = np.reshape(x_test,(x_test.shape[0],-1))

# +

# Define the model and fit data
regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train.ravel())

# +
from statsmodels.tools.eval_measures import mse

# Predict on the test set and compute MSE
y_pred = regressor.predict(x_test)
y_pred = np.reshape(y_pred,(y_pred.size,1))
y_pred = scaler.inverse_transform(y_pred)

y_test = scaler.inverse_transform(y_test)

print(mse(y_pred,y_test))

# +
import matplotlib.pyplot as plt

# Compare Actual vs Predicted for test set
plt.plot(y_test, color = 'blue', label = 'Actual Positive Ratio')
plt.plot(y_pred, color = 'red', label = 'Predicted Positive Ratio')
plt.title('SVR (RBF) Model Actual vs Predicted Comparisson on Test Set')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()

# +
# Compare Actual vs Predicted for all data
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
plt.xlabel('Days after 26/04/2020')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()
