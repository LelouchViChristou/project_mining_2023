# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas dataframe
df = pd.read_csv('data.csv')

# Print statistics
print(df.describe().round(2))
for col in df.columns:
    if col != 'Entity' and col != 'Date':
        plt.hist(df[col], bins=20)
        plt.title(col)
        plt.show()

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Compute and show correlation matrix heatmap
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()
