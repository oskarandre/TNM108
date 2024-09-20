
# ************ Role of Dendrograms for Hierarchical Clustering *********
# import numpy as np
# X = np.array([[5,3],
# [10,15],
# [15,12],
# [24,10],
# [30,30],
# [85,70],
# [71,80],
# [60,78],
# [70,55],
# [80,91] ])

# #Plot
# import matplotlib.pyplot as plt
# labels = range(1, 11)
# plt.figure(figsize=(10, 7))
# plt.subplots_adjust(bottom=0.1)
# plt.scatter(X[:,0],X[:,1], label='True Position')
# for label, x, y in zip(labels, X[:, 0], X[:, 1]):
#  plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
# plt.show()

# # Draw dendrograms
# from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import pyplot as plt
# linked = linkage(X, 'single')
# labelList = range(1, 11)
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#  orientation='top',
#  labels=labelList,
#  distance_sort='descending',
#  show_leaf_counts=True)
# plt.show()


#*********** Hierarchical Clustering via Scikit-Learn***********

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# X = np.array([[5,3],
# [10,15],
# [15,12],
# [24,10],
# [30,30],
# [85,70],
# [71,80],
# [60,78],
# [70,55],
# [80,91] ])

# from sklearn.cluster import AgglomerativeClustering
# cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
# cluster.fit_predict(X)

# plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
# plt.show() 

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage 

customer_data = pd.read_csv('/Users/oskar/OneDrive - Linköpings universitet/Maskininlärning/shopping_data.csv')

print(customer_data.shape)
print(customer_data)

data = customer_data.iloc[:,3:5].values
print(data.shape)
print(data)

# *** DENOGRAMS ***
linked = linkage(data, 'ward')
labelList = range(1, 201)
plt.figure(figsize=(10, 7))
dendrogram(linked,
 orientation='top',
 labels=labelList,
 distance_sort='descending',
 show_leaf_counts=True)
plt.show()

# *** AgglomerativeClustering ***

cluster = AgglomerativeClustering(n_clusters=7, metric='euclidean', linkage='single')
cluster.fit_predict(data)

plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show() 
