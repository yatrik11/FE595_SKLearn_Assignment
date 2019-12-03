# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:11:54 2019

@author: Yatri Kalathia
"""

#SKLearn assignment Question 2

import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

#loading iris dataset
iris_dataset = datasets.load_iris()

#converting to pandas dataframe
iris_data = pd.DataFrame(iris_dataset.data)

iris_features = iris_data.iloc[:, [0,1, 2, 3]].values

no_of_clust = []

#creating n number of CLusters
for i in range(1, 13):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(iris_features)
    no_of_clust.append(kmeans.inertia_)
 
    
for item in range(len(no_of_clust)-1):
    print(abs(no_of_clust[item+1] - no_of_clust[item]))    

    
#Plotting the results onto a line graph to observe elbow heuristic
plt.plot(range(1, 13), no_of_clust)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Values within Cluster Sum of Squares') 
plt.show()


