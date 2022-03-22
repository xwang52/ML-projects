import pandas as pd
import numpy as np
import numpy.linalg as la
import warnings
warnings.filterwarnings('ignore')

"""
The following is a K-means clustering algorithm for data training and testing given several commonly-used 
hyperparameters. K-means is an unsupervised ML technique to group examples into clusters in a way such that 
the total distance of examples to their corresponding centroids are minimized. The number of centroids k is 
pre-determined and the centroids are randomly initialized from the training sample (other initialization 
techniques are also used in scikit-learn for fast convergence). Each example is assigned to the centroid that 
is closest to it in terms of pre-determined distance metric (Manhattan, Euclidean, Minkowksi, etc.). Then the 
centroids are updated by averaging all examples that belong to a given centroid. The process is continued 
iteratively until the centroids do not get updated any more or the number of iteration is reached.  

This implementaion aims at replicating the approach used in scikit-learn, and no specific optimization is used, 
so the algorithm will be slower than scikit-learn. The Kmeans() class defines a few hyperparameters, training 
outputs (final centroids and clustered lables) as well as two functions used for model training and prediction. 
"""

class Kmeans():
  def __init__(self, k, n_iter = 300, metric = 'Euclidean', p = None, 
               tol = 1e-4, seed = 123):
    """
    k: number of centroids
    n_iter: number of iterations used
    metric: distance metric, can be Manhattan, Euclidean, Minkowski
    p: parameter for Minkowski norm
    tol: tolerance level to determine convergence
    seed: seed used for initializing centroids
    """
    self.clusters_ = None
    self.centroids_ = None
    self.num_clusters = k
    self.dist_metric = metric
    self.p = p
    self.n_iter = n_iter
    self.tol = tol
    self.seed = seed

  def fit(self, df):
    n, m = df.shape
    np.random.seed(self.seed)
    centroids = df.loc[np.random.choice(df.index, self.num_clusters)].values
    df['Clusters'] = 0
    if self.dist_metric == 'Manhattan': 
      q = 1
    elif self.dist_metric == 'Euclidean':
      q = 2
    else:
      assert self.p != None, 'Error: Value of p should be provided for Minkowski Distance!'
      q = self.p
    for N in range(self.n_iter):
      for i in range(n):
        """
        broadcast each example to the centroids, and use linalg.norm to calculate the distance
        of each example to the centroids, and finally find the cluster label by using argmin()
        """
        label = la.norm(centroids - df.iloc[i,:-1].values, ord = q, axis = 1).argmin()
        df.iloc[i,-1] = label 
      updated_centroids = df.groupby('Clusters').mean().reset_index().iloc[:,1:].values
      """
      in scikit-learn, frobenius norm of difference in the centroids of two consecutive iterations is evaluated,
      here, we scale the difference by the frobenius norm of centroids of the previous iteration to eliminate the 
      effect of units. 
      """
      if la.norm(centroids - updated_centroids)/la.norm(centroids) < self.tol:
        break
      centroids = updated_centroids

    self.clusters_ = df['Clusters'].values
    self.centroids_ = updated_centroids
    df.drop('Clusters', axis = 1, inplace = True)

  def predict(self, df):
    n, m = df.shape
    df['Clusters'] = 0
    centroids = self.centroids_
    if self.dist_metric == 'Manhattan': 
      q = 1
    elif self.dist_metric == 'Euclidean':
      q = 2
    else:
      q = self.p
    for i in range(n):
      label = la.norm(centroids - df.iloc[i,:-1].values, ord = q, axis = 1).argmin()
      df.iloc[i,-1] = label 
    y = df['Clusters'].values
    df.drop('Clusters', axis = 1, inplace = True)
    return y

# Test the algorithm using normal blobs data with 3 centers and 2 features.
from sklearn import datasets

blobs = datasets.make_blobs(n_samples = 10000, centers = 3, n_features = 2, random_state = 123)
X = pd.DataFrame(blobs[0])
np.random.seed(123)
indices = np.random.permutation(X.shape[0])
ratio = 0.75
train_idx, test_idx = indices[:round(X.shape[0]*ratio)], indices[round(X.shape[0]*ratio):]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

# Fit training data
my_cluster = Kmeans(k = 3)
%time my_cluster.fit(X_train)

# Predict testing data
%time my_cluster.predict(X_test)

# Fit training data using scikit-learn algorithm
from sklearn.cluster import KMeans

sk_cluster = KMeans(n_clusters = 3)
%time sk_cluster.fit(X_train)

# Predict testing data using scikit-learn algorithm
%time sk_cluster.predict(X_test)

# Plot clusters to check consistency
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 10, 8
plt.subplot(221)
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c = my_cluster.clusters_)
plt.title('Train: My Cluster')
plt.subplot(222)
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c = sk_cluster.labels_)
plt.title('Train: Sklearn Cluster')
plt.subplot(223)
plt.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c = my_cluster.predict(X_test))
plt.title('Test: My Cluster')
plt.subplot(224)
plt.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c = sk_cluster.predict(X_test))
plt.title('Test: Sklearn Cluster')
plt.show()
