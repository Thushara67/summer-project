
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from numpy import linalg as LA
from scipy import stats
# import kmeans

iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target
n=X.shape[0]
A = kneighbors_graph(X, 10, mode='connectivity')
Adjacent=A.toarray()

Degree=np.zeros((n,n))
for i in range(n):
    Degree[i,i]=np.sum(A[i,:])

Laplacian = Degree-Adjacent

eigValue, eigVector = LA.eig(Laplacian)
eigVector = eigVector[:,np.argsort(eigValue)]

data = eigVector[:,1:3]
# kmeans.kmeans(data)
# kmeans.kmeans(data)
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(eigVector[:,1:3])
# colors = kmeans.labels_
#
from sklearn.metrics.cluster import adjusted_rand_score
acc=adjusted_rand_score(y,colors)
print "Rand Index is ",acc
