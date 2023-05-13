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
import numpy as np
from scipy.stats import boxcox, yeojohnson
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# read the CSV file into a pandas dataframe
#df = pd.read_csv('data.csv')

# save the updated dataframe as a CSV file
#df.to_csv(('data.csv'), index=True)

df = pd.read_csv('data.csv')
# Compute the new cases and add a new column to the DataFrame
#df['New Deaths'] = grouped_data['Deaths'].di:f()

df['Positive Ratio'] = (df['Cases'].diff() / df['Daily tests']) * 100
df['Death Ratio'] = (df['Deaths'] / df['Cases']) * 100
df["Tested Ratio"]=(df['Daily tests'] / df['Population']) * 100
#df['Deaths per Capita'] = (df['Deaths'] / df['Population'])

#df["Date"] = pd.to_datetime(df["Date"])
df = df.groupby('Entity', group_keys=False).apply(lambda x: x.fillna(method='ffill'))
df = df.groupby('Entity', group_keys=False).apply(lambda x: x.fillna(method='bfill'))
#df = df[df.Cases > 0]

df.drop_duplicates(inplace=True)
#df.dropna(inplace=True)

# Group the data by entity
grouped_data = df.groupby('Entity')

# Compute the average positive ratio, death ratio, tested ratio, and keep the latest values for all other fields
result_data = grouped_data.agg({
    'Continent': 'last',
    'Latitude': 'last',
    'Longitude': 'last',
    'Average temperature per year': 'last',
    'Hospital beds per 1000 people': 'last',
    'GDP/Capita': 'last',
    'Population': 'last',
    'Median age': 'last',
    'Date': 'max',
    'Cases': 'last',
    'Deaths': 'last',
    'Positive Ratio': 'median',
    'Death Ratio': 'median',
    'Tested Ratio': 'median'
    # 'Deaths per Capita': 'last'
})

# Get the unique values in the continent column
continent_values = result_data['Continent'].unique()

# Create a dictionary that maps each unique value to a unique number
continent_map = {continent_values[i]: i+1 for i in range(len(continent_values))}

# Replace the unique values with the corresponding numbers
result_data['Continent'] = result_data['Continent'].replace(continent_map)
result_data.pop('Date')

features = ['Cases', 'Deaths', 'Positive Ratio', 'Death Ratio', 'Tested Ratio'
        # 'Deaths per Capita'
        ]

result_data[features] = result_data[features].astype(float)

scaler = StandardScaler()
normalized_data = result_data.loc[:,(features)]
normalized_data[features] = scaler.fit_transform(result_data[features])

# n_clusters=10
# # Cluster the data using the best k value
# kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init="auto")
# kmeans.fit(normalized_data)
# normalized_data['Cluster'] = kmeans.labels_
# result_data['Cluster'] = kmeans.labels_

# # Plot the clusters
# labels = kmeans.labels_
# scatter = plt.scatter(normalized_data['Death Ratio'], normalized_data['GDP/Capita'], c=labels)

# plt.title('Clusters')
# plt.xlabel("Death Ratio")
# plt.ylabel("GDP/Capita")

# handles, labels = scatter.legend_elements()
# legend_labels = [f'Cluster {i}' for i in range(n_clusters)]
# legend = plt.legend(handles, legend_labels, loc='upper right', title='Clusters')

# cluster_groups = result_data.groupby('Cluster')
# cluster_entity_names = cluster_groups.apply(lambda x: ', '.join(x.index))
# cluster_entity_names_df = pd.DataFrame({'Entity Names': cluster_entity_names})
# print(cluster_entity_names_df.to_string())



# Create a dendrogram to help determine the optimal number of clusters
#dendrogram = sch.dendrogram(sch.linkage(normalized_data, method='average'))
dendrogram = sch.dendrogram(sch.linkage(normalized_data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Entities')
plt.ylabel('Euclidean distances')
plt.axhline(y = 15.25, color = 'r', linestyle = '-')
plt.axhline(y = 10.75, color = 'r', linestyle = '--')
plt.show()

#distance_threshold = 7.0
#cluster = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='ward', distance_threshold=distance_threshold)
cluster = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
cluster.fit_predict(normalized_data)
normalized_data['Cluster'] = cluster.labels_
result_data['Cluster'] = cluster.labels_

#grouped_df = df.groupby('Entity', group_keys=False)
#for key, item in grouped_df:
#    print(grouped_df.get_group(key), "\n\n")


# +

# Plot the clusters
#labels = cluster.labels_
#scatter = plt.scatter(result_data['Death Ratio'], result_data['GDP/Capita'], c=labels)

#plt.title('Clusters')
#plt.xlabel("Death Ratio")
#plt.ylabel("GDP/Capita")

#handles, labels = scatter.legend_elements()
#legend_labels = [f'Cluster {i}' for i in range(15)]
#legend = plt.legend(handles, legend_labels, loc='upper right', title='Clusters')

cluster_groups = result_data.groupby('Cluster')
cluster_entity_names = cluster_groups.apply(lambda x: ', '.join(x.index))
cluster_entity_names_df = pd.DataFrame({'Entity Names': cluster_entity_names})
print(cluster_entity_names_df.to_string())

# Compute cluster statistics
cluster_stats = result_data.groupby('Cluster').agg({
    #'Continent': lambda x: x.mode().iloc[0],
    'Continent': ['mean', 'std'],
    'Latitude': ['mean', 'std'],
    'Longitude': ['mean', 'std'],
    'Average temperature per year': ['mean', 'std'],
    'Hospital beds per 1000 people': ['mean', 'std'],
    'GDP/Capita': ['mean', 'std'],
    'Population': ['mean', 'std'],
    'Median age': ['mean', 'std'],
    'Cases': ['mean', 'std'],
    'Deaths': ['mean', 'std'],
    'Positive Ratio': ['mean', 'std'],
    'Death Ratio': ['mean', 'std'],
    'Tested Ratio': ['mean', 'std']
    #'Deaths per Capita': ['mean', 'std']
})

# Print cluster statistics
for cluster_id in range(len(cluster_stats)):
    print(f"\nCluster {cluster_id} statistics:")
    #print(f"Continent: {cluster_stats.loc[cluster_id, ('Continent', '<lambda>')]}")
    print(f"Continent: mean = {cluster_stats.loc[cluster_id, ('Continent', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Continent', 'std')]:.2f}")
    print(f"Latitude: mean = {cluster_stats.loc[cluster_id, ('Latitude', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Latitude', 'std')]:.2f}")
    print(f"Longitude: mean = {cluster_stats.loc[cluster_id, ('Longitude', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Longitude', 'std')]:.2f}")
    print(f"Average temperature per year: mean = {cluster_stats.loc[cluster_id, ('Average temperature per year', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Average temperature per year', 'std')]:.2f}")
    print(f"Hospital beds per 1000 people: mean = {cluster_stats.loc[cluster_id, ('Hospital beds per 1000 people', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Hospital beds per 1000 people', 'std')]:.2f}")
    print(f"GDP/Capita: mean = {cluster_stats.loc[cluster_id, ('GDP/Capita', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('GDP/Capita', 'std')]:.2f}")
    print(f"Population: mean = {cluster_stats.loc[cluster_id, ('Population', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Population', 'std')]:.2f}")
    print(f"Median age: mean = {cluster_stats.loc[cluster_id, ('Median age', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Median age', 'std')]:.2f}")
    print(f"Cases: mean = {cluster_stats.loc[cluster_id, ('Cases', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Cases', 'std')]:.2f}")
    print(f"Deaths: mean = {cluster_stats.loc[cluster_id, ('Deaths', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Deaths', 'std')]:.2f}")#na kanononikopiithi me to population
    print(f"Positive Ratio: mean = {cluster_stats.loc[cluster_id, ('Positive Ratio', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Positive Ratio','std')]:.2f}")
    print(f"Death Ratio: mean = {cluster_stats.loc[cluster_id, ('Death Ratio', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Death Ratio', 'std')]:.2f}")
    print(f"Tested Ratio: mean = {cluster_stats.loc[cluster_id, ('Tested Ratio', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Tested Ratio', 'std')]:.2f}")
    #print(f"Deaths per Capita: mean = {cluster_stats.loc[cluster_id, ('Deaths per Capita', 'mean')]:.2f}, std = {cluster_stats.loc[cluster_id, ('Deaths per Capita', 'std')]:.2f}")#na kanononikopiithi me to population
