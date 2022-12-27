import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
%matplotlib inline


df = pd.read_csv('Dados_Python_V3.csv')

#Creating the dataset used to apply K-Means and find the clusters, so we can determinate the Training, Validation and
#Testing datasets. The variables used were the dry bulb air temperature entering the tower and the temperature
#of the cold water leaving the tower.

base_teste = pd.DataFrame(df, columns=['T_bulbo_seco_ar', 'T_agua_fria'])


#Applying K-Means

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(base_teste)

#Visualizing the clusters

import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=base_teste, x="T_bulbo_seco_ar", y="T_agua_fria", hue=kmeans.labels_)
plt.show()

kmeans.labels_.shape

pd.DataFrame(kmeans.labels_).to_csv('KMeans_Labels.csv')

