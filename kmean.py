#!/usr/bin/python
# -*- coding: utf-8 -*-

from random import *
import matplotlib.pyplot as plt
import random
import math
import time
from sklearn import datasets
import numpy as np
from numpy.random import randint
from scipy.special import comb
from sklearn.metrics.cluster import adjusted_rand_score

maxfloat = float('inf')
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target


# final cluster plot made using 3 clusters
# print(y)

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
        if len(multi_arr2[row]) != 0:
            for col in range(len(multi_arr2[row])):
                ele = multi_arr2[row][col]
                vector1 = multi_arr1[ele]
                res_list1 = [sum(j) for j in zip(res_list1, vector1)]

            res_list1 = [x / len(multi_arr2[row]) for x in res_list1]

            for i in range(col_num):
                res_list[row].append(res_list1[i])
        elif len(multi_arr2[row]) == 0:
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


def confusion_matrix(list1, list2, cluster_num):
    list3 = [[0 for col in range(3)] for row in range(3)]
    for i in range(cluster_num):
        for j in range(len(list1[i])):
            if list2[list1[i][j]] == 0:
                list3[i][0] = list3[i][0] + 1
            elif list2[list1[i][j]] == 1:
                list3[i][1] = list3[i][1] + 1
            elif list2[list1[i][j]] == 2:
                list3[i][2] = list3[i][2] + 1
    return list3


# multi_list1 - contains datapoints
# multi_list2 - contains centroids for respective clusters
# cluster_list - contains datapoints withrespect to each cluster

def kmeans(X, col_num):
    row_num = 150  # n = 150

  # 2

    cluster_num = 3  # k

# generating datapoints

    multi_list1 = [[random.uniform(1, 1000) for col in range(col_num)]
                   for row in range(row_num)]

    multi_list1 = X

    multi_list = [[float(0) for col in range(col_num)] for row in
                  range(cluster_num)]  # contains points
    list_1 = [multi_list1[i][0] for i in range(row_num)]
    list_2 = [multi_list1[i][1] for i in range(row_num)]

    choices = list(range(row_num - 1))
    random.shuffle(choices)

    # used to randomly allot representative(centroids) from data points

    mean_arr = [choices.pop() for i in range(cluster_num)]

    multi_list2 = [[float(0) for col in range(col_num)] for row in
                   range(cluster_num)]  # (centroids)representatives of respective clusters

    for row in range(cluster_num):
        i = mean_arr.pop()
        for col in range(col_num):
            multi_list2[row][col] = multi_list1[i][col]

# kmeans++

    plt.figure(figsize=(10, 5))

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

        multi_list3 = centroid(multi_list1, cluster_list, multi_list2,
                               cluster_num, col_num)

        for row in range(cluster_num):
            vector1 = multi_list2[row]
            vector2 = multi_list3[row]

            error = error + euclidean(multi_list2[row],
                    multi_list3[row])

    # print(error/cluster_num)

        multi_list2 = multi_list3

# print('centroids')
# print(multi_list2)

    end = time.time()
    print ('time taken by centroids to converge')
    print (end - start)
    print ('no. of iterations of while loop')
    print (count)

    v = [0 for i in range(row_num)]
    for i in range(3):
        for j in range(len(cluster_list[i])):
            v[cluster_list[i][j]] = i

    return v;

