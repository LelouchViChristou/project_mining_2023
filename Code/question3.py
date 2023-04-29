# +
import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df = pd.read_csv('data.csv')

greece_df = df.loc[df['Entity'] == 'Greece', ['Date', 'Cases', 'Deaths', 'Daily tests']].copy()
greece_df.dropna(inplace=True)
greece_df["Date"] = pd.to_datetime(greece_df["Date"])
greece_df['New Cases'] = greece_df['Cases'].diff()
greece_df = greece_df.fillna(0)
greece_df['Positive Ratio'] = (greece_df['New Cases'] / greece_df['Daily tests']) * 100
greece_df['Death Ratio'] = (greece_df['Deaths'] / greece_df['Cases']) * 100
greece_df["Tested ratio"] = (greece_df['Daily tests'] / 10760421.0) * 100

# create a MinMax scaler object for all columns
scaler = StandardScaler()

# select all columns to normalize
columns_to_normalize = ['Daily tests','Cases', 'Deaths','New Cases','Positive Ratio','Death Ratio',"Tested ratio"]

# fit and transform all columns with the scaler object
greece_df[columns_to_normalize] = scaler.fit_transform(greece_df[columns_to_normalize])
print(greece_df.head(20).to_string())
