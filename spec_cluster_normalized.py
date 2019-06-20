'''
normalised graph laplacian Lrw ,Lsym


'''

from random import *
import matplotlib.pyplot as plt
import random
import math
import time
from sklearn import datasets
import numpy as np
from numpy.random import randint
from scipy.special import comb
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

maxfloat = float('inf')
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
def euclidean(vector1, vector2):

    dist = [(a - b) ** 2 for (a, b) in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))

    return dist

  	
def normalisation(list1,row_num):
    sum_ = 0
    list2 =[[float(0) for col in
               range(len(list1[0]))] for row in range(row_num)]  
    list2 =list1
    col = len(list1[0])
    for i in range(row_num):
        for j in range( col):
            sum_ = sum_+ (list1[i][j])**2
        list2[i] = np.divide(list1[i],sum_)
       	
    return list2

sigma = 1000




# multi_list10 - contains datapoints

rownum = 150#int(input('Input number of datapoints: '))  # n
colnum = 2  # d
clusternum = 3  # k

multi_list10 = [[float(random.uniform(1, 1000)) for col in
               range(colnum)] for row in range(rownum)]  # contains points
multi_list10 = X

ad_list = [[float(0) for col in range(rownum)] for row in
               range(rownum)]


# --------------------weights(W)

for row in range(rownum):

    for col in range(rownum):
        
        ad_list[row][col] = math.exp(-1*euclidean(multi_list10[row],multi_list10[col])*euclidean(multi_list10[row],multi_list10[col])/(2*sigma*sigma))

#print(ad_list)



#----------------------------degree matrix -- diagonal matrix(D)
degree_list = [[float(0) for col in
               range(rownum)] for row in range(rownum)] 

summ = 0

for row in range(rownum):

    for col in range(rownum):
        summ = ad_list[row][col] + summ
    degree_list[row][row] = summ       

#print(degree_list)

I = [[float(0) for col in
               range(rownum)] for row in range(rownum)]
for row in range(rownum):

    for col in range(rownum):
        summ = 1
    degree_list[row][row] = summ       



#----------------------laplacian matrix = I-D^-1W , I - D-1/2 W D-1/2  


laplacian_list = [[float(0) for col in
               range(rownum)] for row in range(rownum)] 


#---- Lrw

D_inv = inv(degree_list)
L_ = np.matmul(D_inv,ad_list) 
laplacian_list = np.subtract(I,L_)


# --------- normalised graph laplacian Lsym
'''
D_inv = inv(degree_list)
D_sqrt = np.sqrt(D_inv)
laplacian_list = np.subtract(I,np.matmul(np.matmul(D_sqrt,ad_list),D_sqrt))
'''
eigenValues,v = LA.eig(laplacian_list)

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
v= v[:,idx]



# --------------list of eigen vectors

f_list = [[float(random.uniform(1, 1000)) for col in
               range(clusternum)] for row in range(rownum)]  # contains points
'''print(multi_list1)
'''


'''
f_list = normalisation(f_list,rownum)
'''


for row in range(rownum):
    for col in range(clusternum):
        f_list[row][col] =v[row][rownum-1-col]



# ------------------ kmeans
X_ = np.array(f_list)
X_ = X_.real

plt.figure(figsize = (10,5))
kmeans = KMeans(n_clusters=3)  
kmeans.fit(X_)

plt.subplot(121)
plt.title('spectral clustered')
plt.scatter(multi_list10[:,0],multi_list10[:,1], c=kmeans.labels_, cmap='rainbow')
plt.subplot(122)
plt.title('actual')
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
print("rand score")
print(adjusted_rand_score(y,kmeans.labels_))

plt.show()

