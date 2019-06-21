
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from numpy import linalg as LA
import kmeans2

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
kmeans2.kmeans(data,y,X)
