import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from numpy import linalg as la
import time
import sklearn
from scipy import stats
from sklearn import datasets

def kmeans(data,labels,X):
    k=3
# f = open('iris.data', 'r+')
# data=np.loadtxt(f,delimiter=',', usecols=(0,1,2,3))
# f.close()
    n=data.shape[0]
    dims=data.shape[1]
    cum_sum = np.zeros(n)
# c is our set of centroids
    c = np.zeros((k, dims))
    prev_c = np.zeros((k, dims))

# randomly choose some point to initialise our first centroid
    random_point = np.random.choice(n)
    c[0,:] = data[random_point,:]

# function to calculate eucildean distance between two points
    def distance(x, y):
        a = la.norm(np.subtract(x,y))
        return a

# min_dist finds the distance between the given point x and the centre which is closest to it
    def min_dist(x,c):
        min=distance(c[0,:],x)
        for count, ele in enumerate(c):
            d=distance(x,ele)
            if d<min :
                min=d
        return min


    start = time.time()

# the following code is to assign centres
    for x in range(1,k):

	# cum_sum holds the cumulative sum of the squares of D where D= distance between  the point and its closest centre
        prev=0
        for i in range(n):
            cum_sum[i] = np.square(min_dist(data[i,:],c)) + prev
            prev = cum_sum[i]

        total = prev
	# with the next step, we make sure cum_sum holds only values from 0 to 1
        cum_sum = cum_sum/total
	# "random" is a random number between 0 and 1
        random = np.random.random()
        for i in range(n):
            if cum_sum[i]>random:
                c[x,:]=data[i,:]
                break


    end = time.time()
    plt.figure(figsize = (12,4))

    dist = np.zeros(k)
    classify=np.zeros(n)

    error=la.norm(c-prev_c)


    while error>0.0001 :
    #the following loop creates a n*1 array which stores the centroid each point is closest to
        for i in range(n):
            for l in range(k):
                dist[l] = distance(data[i,:],c[l,:])
                classify[i] = np.argmin(dist)

        prev_c=c.copy()

    # the following code updates the centroids by taking means of each cluster
        for i in range(k):
            d=0
            a = np.zeros(dims)
            for j in range(n):
                if classify[j]==i:
                    a=a+data[j,:]
                    d=d+1
            c[i,:] =a/d

        error=la.norm(c-prev_c)

# classifying points

    dist = np.zeros(k)
    classify=np.zeros(n)

    for i in range(n):
        for l in range(k):
            dist[l] = distance(data[i,:],c[l,:])
            classify[i] = np.argmin(dist)
    plt.scatter(X[:,0], X[:,1], c= classify, s=5, alpha = 0.3)
    # plt.scatter(c[:,0], c[:,1], marker='*', c='green', s=150)
    # plt.show()

    # plt.scatter(data[:,0],data[:,1])
    plt.show()
    from sklearn.metrics.cluster import adjusted_rand_score

    acc=adjusted_rand_score(labels,classify)
    print "Rand Index is ",acc
