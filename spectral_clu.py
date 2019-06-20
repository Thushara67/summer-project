
'''

I used unnormalised graph laplacian



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


def myComb(a,b):
  return comb(a,b,exact=True)

vComb = np.vectorize(myComb)

# --- rand index evaluation

def get_tp_fp_tn_fn(cooccurrence_matrix):
  tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int),2).sum()
  tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int),2).sum()
  tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
  fp = tp_plus_fp - tp
  fn = tp_plus_fn - tp
  tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

  return [tp, fp, tn, fn]

maxfloat = float('inf')
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
#print(len(y))
# final cluster plot made using 3 clusters
#print(y)


def euclidean(vector1, vector2):

    dist = [(a - b) ** 2 for (a, b) in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))

    return dist

def centroid(
    multi_arr1,
    multi_arr2,
    multi_arr3,
    cluster_num,
    col_num,
    ):

    res_list1 = [float(0) for i in range(col_num)]
    res_list = []
    for j in range(cluster_num):
        res_list.append([])
    for row in range(cluster_num):
        res_list1 = [float(0) for i in range(col_num)]
        if len(multi_arr2[row])!=0:
           for col in range(len(multi_arr2[row])):
               ele = multi_arr2[row][col]
               vector1 = multi_arr1[ele]
               res_list1 = [sum(j) for j in zip(res_list1, vector1)]
              
           res_list1 = [x / len(multi_arr2[row]) for x in res_list1]

           for i in range(col_num):
               res_list[row].append(res_list1[i])
        elif len(multi_arr2[row])== 0 :
           for i in range(col_num):
               res_list[row].append(multi_arr3[row][i])             

    return res_list

# finds min distance between a point and centroid

def mindist(array, centroid, k):
    min_dist = maxfloat
    for i in range(k):
        dist = euclidean(array, centroid[i])
        if min_dist > dist:
            min_dist = dist
    return min_dist

def evaluate(list1,list2,cluster_num):
    list3 =[[0 for col in range(3)]
               for row in range(3)]
    for i in range(cluster_num):
        for j in range(len(list1[i])):
            if list2[list1[i][j]] == 0:
               list3[i][0] = list3[i][0] + 1
            elif list2[list1[i][j]] == 1:
                 list3[i][1] = list3[i][1] + 1
            elif list2[list1[i][j]] ==2:
                 list3[i][2] = list3[i][2] + 1
    return list3
               

sigma = 1000

# multi_list0 - contains datapoints

rownum = 150  # n
colnum = 2  # d
clusternum = 3  # k

multi_list0 = [[float(random.uniform(1, 1000)) for col in
               range(colnum)] for row in range(rownum)]  # contains points
multi_list0 = X

# adjacency matrix(A)

ad_list = [[float(0) for col in range(rownum)] for row in
               range(rownum)]

# weights

for row in range(rownum):

    for col in range(rownum):
       
        ad_list[row][col] = math.exp(-1*euclidean(multi_list0[row],multi_list0[col])*euclidean(multi_list0[row],multi_list0[col])/(2*sigma*sigma))


#degree matrix -- diagonal matrix(D)

degree_list = [[float(0) for col in
               range(rownum)] for row in range(rownum)] 

summ = 0

for row in range(rownum):

    for col in range(rownum):
        summ = ad_list[row][col] + summ
    degree_list[row][row] = summ       

#print(degree_list)

#laplacian matrix = D-A
laplacian_list = [[float(0) for col in
               range(rownum)] for row in range(rownum)] 
for row in range(rownum):

    for col in range(rownum):
       
        laplacian_list[row][col] = degree_list[row][col] - ad_list[row][col]

eigenValues,eigenvec = LA.eig(laplacian_list)

# -- eigen values are sorted into descending order

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenvec= eigenvec[:,idx]

#--contains first k eigen vectors

f_list = [[float(random.uniform(1, 1000)) for col in
               range(clusternum)] for row in range(rownum)]  

for row in range(rownum):
    for col in range(clusternum):
        f_list[row][col] =eigenvec[row][rownum-1-col]

# ------------------ kmeans

# multi_list1 - contains datapoints on which kmeans will be applied
# multi_list2 - contains centroids for respective clusters
# cluster_list - contains datapoints withrespect to each cluster

row_num = rownum  # n = 150
col_num = clusternum  # 2
cluster_num = 3 # k
# generating datapoints


multi_list1 = [[random.uniform(1, 1000) for col in range(col_num)]
               for row in range(row_num)]

multi_list1 = f_list

multi_list = [[float(0) for col in
               range(col_num)] for row in range(cluster_num)]  # contains points

list_1 = [multi_list0[i][0] for i in range(row_num)]
list_2 = [multi_list0[i][1] for i in range(row_num)]
#print("multi_list1")
#print(multi_list1)

choices = list(range(row_num - 1))
random.shuffle(choices)

# used to randomly allot representatives(centroids) from data points

mean_arr = [choices.pop() for i in range(cluster_num)]

multi_list2 = [[float(0) for col in range(col_num)] for row in
               range(cluster_num)]  # (centroids)representatives of respective clusters
for row in range(cluster_num):
    i = mean_arr.pop()
    for col in range(col_num):
        multi_list2[row][col] = multi_list1[i][col]
'''
print (multi_list2)
'''
plt.figure(figsize = (10,5))





error = 2
count = 0
start = time.time()
while error > 0.005:
    cluster_list = []
    
    for j in range(cluster_num):
        cluster_list.append([])
    error = 0
    count = count + 1

    

	
    for row in range(row_num):

        vector1 = multi_list1[row]
        min_dist = maxfloat
        index = -1
        for num in range(cluster_num):
            vector2 = multi_list2[num]

            dist = euclidean(vector1, vector2)
            if min_dist > dist:
                min_dist = dist
                index = num

        cluster_list[index].append(row)

    multi_list3 = centroid(multi_list1, cluster_list,multi_list2, cluster_num,
                           col_num)
    
    for row in range(cluster_num):
        vector1 = multi_list2[row]
        vector2 = multi_list3[row]

        error = error + euclidean(multi_list2[row], multi_list3[row])
    
    
    multi_list2 = multi_list3


    if count == 2:
    	multi_list = multi_list2

'''   
print('centroids')
print(multi_list2)
'''
end = time.time()
print('time taken by centroids to converge')
print (end - start)
print ('no. of iterations of while loop')
print (count) 
# evaluation
list_10 = evaluate(cluster_list,y,3)
print(list_10)  
list10 = np.matrix(list_10)


list_0 =get_tp_fp_tn_fn(list10)

sum_ = ((list_0)[0] +list_0[2])/(sum(list_0))
print("rand_index")
print(sum_)

list1 =  [0 for i in range(len(cluster_list[0]))] 
list2 =  [0 for i in range(len(cluster_list[0]))]  

for j in range(len(cluster_list[0])):
         list1[j] = multi_list0[cluster_list[0][j]][0]
         list2[j] = multi_list0[cluster_list[0][j]][1]

list3 = [0 for i in range(len(cluster_list[1]))]
list4 = [0 for i in range(len(cluster_list[1]))]
for j in range(len(cluster_list[1])):
         list3[j] = multi_list0[cluster_list[1][j]][0]
         list4[j] = multi_list0[cluster_list[1][j]][1]     

list5 =  [0 for i in range(len(cluster_list[2]))] 
list6 =  [0 for i in range(len(cluster_list[2]))]   

for j in range(len(cluster_list[2])):
         list5[j] = multi_list0[cluster_list[2][j]][0]
         list6[j] = multi_list0[cluster_list[2][j]][1]  

#list7 = [multi_list2[i][0] for i in range(cluster_num)]
#list8 = [multi_list2[i][1] for i in range(cluster_num)]



plt.subplot(121)
plt.title('final clusters - predicted')


plt.scatter(list1, list2,label = "stars", color= "red",  
            marker= "o")
#list_5 = [multi_list[i][0] for i in range(cluster_num)]
#list_6 = [multi_list[i][1] for i in range(cluster_num)]
plt.ylim(1,5) 
plt.xlim(4,9)     
plt.scatter(list3, list4, label= "stars", color= "blue",  
            marker= "o")


plt.scatter(list5, list6, label= "stars", color= "black",  
            marker= "o")


#plt.scatter(list7, list8, label= "stars", color= "brown",  
#  marker= "*", s = 100)

plt.subplot(122)
plt.title('actual')
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')



plt.show()

