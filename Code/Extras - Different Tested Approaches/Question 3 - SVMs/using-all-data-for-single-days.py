# +
import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

df = pd.read_csv('../../data.csv')

greece_df = df.loc[df['Entity'] == 'Greece', ['Date', 'Cases', 'Deaths', 'Daily tests']].copy()
greece_df = greece_df.reset_index(drop=True)
greece_df = greece_df.drop(greece_df.index[range(0,60)])
greece_df = greece_df.reset_index(drop=True)

greece_df['Positive Ratio'] = (greece_df['Cases'].diff() / greece_df['Daily tests']) * 100
greece_df['Death Ratio'] = (greece_df['Deaths'] / greece_df['Cases']) * 100
greece_df["Tested Ratio"] = (greece_df['Daily tests'] / 10760421.0) * 100
greece_df.iloc[304,4] = float("NaN") #Outlier
greece_df = greece_df.apply(lambda x: x.fillna(method='ffill'))
greece_df = greece_df.apply(lambda x: x.fillna(method='bfill'))
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
greece_df

# +
x = greece_df[['Cases', 'Deaths', 'Positive Ratio', 'Death Ratio', 'Tested Ratio']]
y = greece_df[['Positive Ratio']]
x = x.tail(-3)
y = y.head(-3)
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

# create a MinMax scaler object for all columns
x_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))
x[['Cases','Deaths','Positive Ratio','Death Ratio','Tested Ratio']] = x_scaler.fit_transform(x[['Cases','Deaths','Positive Ratio','Death Ratio','Tested Ratio']])
y[['Positive Ratio']] = y_scaler.fit_transform(y[['Positive Ratio']])


x_train = x.iloc[:greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0]]
y_train = y.iloc[:greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0]]
x_test = x.iloc[greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0]-3:]
y_test = y.iloc[greece_df.loc[greece_df['Date'] == '2021-01-01'].index[0]-3:]

# +
# Define the model

regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train.values.ravel())

# +
from statsmodels.tools.eval_measures import mse
y_pred = regressor.predict(x_test)
y_pred = np.reshape(y_pred,(y_pred.size,1))
y_pred = y_scaler.inverse_transform(y_pred)

y_test = y_scaler.inverse_transform(y_test)

print(mse(y_pred,y_test))

# +
import matplotlib.pyplot as plt

plt.plot(y_test, color = 'blue', label = 'Actual Positive Ratio')
plt.plot(y_pred, color = 'red', label = 'Predicted Positive Ratio')
plt.title('SVR (RBF) Model Loss:')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()

# +
y_pred = regressor.predict(x)
y_pred = np.reshape(y_pred,(y_pred.size,1))
y_pred = y_scaler.inverse_transform(y_pred)
y_plot = np.reshape(y,(y.size,1))
y_plot = y_scaler.inverse_transform(y_plot)

plt.plot(y_plot, color = 'blue', label = 'Actual Positive Ratio')
plt.plot(y_pred, color = 'red', label = 'Predicted Positive Ratio')
plt.title('Actual vs Predicted Positive Ratio On All Data')
plt.xlabel('Days after 01/01/2021')
plt.ylabel('Positive Ratio')
plt.legend()
plt.show()
