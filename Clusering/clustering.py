from pandas import DataFrame
from sklearn.cluster import KMeans
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


np.random.seed(0)
X=np.random.uniform(10,50,(50,2))


X_train, X_test= sklearn.model_selection.train_test_split(X, test_size = 0.33, random_state = 5)

kmeans = KMeans(n_clusters=4).fit(X_train)
centroids = kmeans.cluster_centers_
prediction = kmeans.predict(X_test)

print(centroids)
print(prediction)