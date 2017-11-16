# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:01:12 2017

@author: Sumit
"""

#import heapq

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from time import time


#from dim_reduce import dim_reduce

#==============================================================================
# def fourier_series(x, *a):
#     """
#     Arbitrary dimensionality fourier series.
# 
#     The first parameter is a_0, and the second parameter is the interval/scale
#     parameter.
# 
#     The parameters are altering sin and cos paramters.
# 
#     n = (len(a)-2)/2
#     """
#     output = 0
#     output += a[0]/2
#     w = a[1]
#     for n in range(2, len(a), 2):
#         n_ = n/2
#         val1 = a[n]
#         val2 = a[n+1]
#         output += val1*np.sin(n_*x*w)
#         output += val2*np.cos(n_*x*w)
#     return output
# 
# def graph_distances(start, edges, distances):
#     """
#     Given an undirected adjacency list and a pairwise distance matrix between
#     all nodes: calculates distances along graph from start node.
# 
#     Args:
#         start (int): start node
#         edges (list): adjacency list of tuples
#         distances (array): 2d array of distances between nodes
# 
#     Returns:
#         dict of node to distance from start
#     """
#     # convert adjacency list to adjacency dict
#     adj = {x: [] for x in range(len(distances))}
#     for n1, n2 in edges:
#         adj[n1].append(n2)
#         adj[n2].append(n1)
#     # run dijkstra's algorithm
#     to_visit = []
#     new_dist = {}
#     for n in adj[start]:
#         heapq.heappush(to_visit, (distances[start, n], n))
#     while to_visit:
#         d, next_node = heapq.heappop(to_visit)
#         if next_node not in new_dist:
#             new_dist[next_node] = d
#         for n in adj[next_node]:
#             if n not in new_dist:
#                 heapq.heappush(to_visit, (d + distances[next_node, n], n))
#     return new_dist
# 
# 
# def poly_curve(x, *a):
#     """
#     Arbitrary dimension polynomial.
#     """
#     output = 0.0
#     for n in range(0, len(a)):
#         output += a[n]*x**n
#     return output
# 
# 
# def lineage(means, weights, curve_function='poly', curve_dimensions=6):
#     """
#     Lineage graph produced by minimum spanning tree
# 
#     Args:
#         means (array): genes x clusters - output of state estimation
#         weights (array): clusters x cells - output of state estimation
#         curve_function (string): either 'poly' or 'fourier'. Default: 'poly'
#         curve_dimensions (int): number of parameters for the curve. Default: 6
# 
#     Returns:
#         curve parameters: list of lists for each cluster
#         smoothed data in 2d space: 2 x cells
#         list of edges: pairs of cell indices
#         cell cluster assignments: list of ints
#     """
#     if curve_function=='poly':
#         func = poly_curve
#     elif curve_function=='fourier':
#         func = fourier_series
#     # step 1: dimensionality reduction
#     X = dim_reduce(means, weights, 2)
#     reduced_data = np.dot(X.T, weights)
#     if X.shape[0]==2:
#         reduced_data = np.dot(X, weights)
#     # 2. identifying dominant cell types - max weight for each cell
#     cells = weights.shape[1]
#     clusters = means.shape[1]
#     cell_cluster_assignments = weights.argmax(0)
#     # 3. fit smooth curve over cell types -5th order fourier series
#     # cluster_curves contains the parameters for each curve.
#     cluster_curves = []
#     # cluster_fitted_vals is a 2 x cells array
#     cluster_fitted_vals = reduced_data.copy()
#     # cluster_edges contain a list of ordered pairs (indices) connecting cells
#     # in each cluster.
#     cluster_edges = []
#     for c in range(clusters):
#         cluster_cells = reduced_data[:, cell_cluster_assignments==c]
#         if len(cluster_cells) == 0:
#             cluster_edges.append([])
#             continue
#         if cluster_cells.shape[1] < 2:
#             cluster_edges.append([])
#             continue
#         elif cluster_cells.shape[1] < curve_dimensions:
#             tc = cluster_cells.shape[1]-1
#         else:
#             tc = curve_dimensions
#         # y = f(x)
#         if curve_function=='fourier':
#             p0 = [1.0]*tc
#             # scipy is bad at finding the correct scale
#             p0[1] = 0.0001
#             bounds = (-np.inf, np.inf)
#         else:
#             p0 = [1.0]*tc
#             bounds = (-np.inf, np.inf)
#         p_x, pcov_x = curve_fit(func, cluster_cells[0,:],
#                 cluster_cells[1,:],
#                 p0=p0, bounds=bounds)
#         perr_x = np.sum(np.sqrt(np.diag(pcov_x)))
#         # x = f(y)
#         p_y, pcov_y = curve_fit(func, cluster_cells[1,:],
#                 cluster_cells[0,:],
#                 p0=p0, bounds=bounds)
#         perr_y = np.sum(np.sqrt(np.diag(pcov_y)))
#         if perr_x <= perr_y:
#             x_vals = reduced_data[0,:]
#             cluster_curves.append(p_x)
#             y_vals = np.array([func(x, *p_x) for x in x_vals])
#             #print 'error:', np.sum(np.sqrt((y_vals - reduced_data[1,:])**2)[cell_cluster_assignments==c])
#             fitted_vals = np.array([x_vals, y_vals])
#             cluster_fitted_vals[:,cell_cluster_assignments==c] = fitted_vals[:,cell_cluster_assignments==c]
#             # sort points by increasing X, connect points
#             x_indices = np.argsort(x_vals)
#             x_indices = [x for x in x_indices if cell_cluster_assignments[x]==c]
#             new_cluster_edges = []
#             for i, j in zip(x_indices[:-1], x_indices[1:]):
#                 new_cluster_edges.append((i,j))
#             cluster_edges.append(new_cluster_edges)
#         else:
#             y_vals = reduced_data[1,:]
#             cluster_curves.append(p_y)
#             x_vals = np.array([func(x, *p_y) for x in y_vals])
#             #print 'error:', np.sum(np.sqrt((x_vals - reduced_data[0,:])**2)[cell_cluster_assignments==c])
#             fitted_vals = np.array([x_vals, y_vals])
#             cluster_fitted_vals[:,cell_cluster_assignments==c] = fitted_vals[:,cell_cluster_assignments==c]
#             # sort points by increasing Y, connect points
#             y_indices = np.argsort(y_vals)
#             y_indices = [x for x in y_indices if cell_cluster_assignments[x]==c]
#             new_cluster_edges = []
#             for i,j in zip(y_indices[:-1], y_indices[1:]):
#                 new_cluster_edges.append((i,j))
#             cluster_edges.append(new_cluster_edges)
#     # 4. connect each cluster together
#     # for each cluster, find the closest point in another cluster, and connect
#     # those points. Add that point to cluster_edges.
#     # build a distance matrix between the reduced points...
#     distances = squareform(pdist(cluster_fitted_vals.T))
#     for c1 in range(clusters):
#         min_dist = np.inf
#         min_index = None
#         if sum(cell_cluster_assignments==c1)==0:
#             continue
#         for c2 in range(clusters):
#             if sum(cell_cluster_assignments==c2)==0:
#                 continue
#             if c1!=c2:
#                 distances_c = distances[cell_cluster_assignments==c1,:][:, cell_cluster_assignments==c2]
#                 mindex = np.unravel_index(distances_c.argmin(), distances_c.shape)
#                 if distances_c[mindex] < min_dist:
#                     min_dist = distances_c[mindex]
#                     min_index = np.where(distances==min_dist)
#                     min_index = (min_index[0][0], min_index[1][0])
#         cluster_edges[c1].append(min_index)
#     # flatten cluster_edges?
#     cluster_edges = [i for sublist in cluster_edges for i in sublist]
#     return cluster_curves, cluster_fitted_vals, cluster_edges, cell_cluster_assignments
# 
# def pseudotime(starting_node, edges, fitted_vals):
#     """
#     Args:
#         starting_node (int): index of the starting node
#         edges (list): list of tuples (node1, node2)
#         fitted_vals (array): output of lineage (2 x cells)
# 
#     Returns:
#         A 1d array containing the pseudotime value of each cell.
#     """
#     # TODO
#     # 1. calculate a distance matrix...
#     distances = np.array([[sum((x - y)**2) for x in fitted_vals.T] for y in fitted_vals.T])
#     # 2. start from the root node/cell, calculate distance along graph
#     distance_dict = graph_distances(starting_node, edges, distances)
#     output = []
#     for i in range(fitted_vals.shape[1]):
#         output.append(distance_dict[i])
#     return np.array(output)
#==============================================================================


#This function uses cKDTree to find the nearest points in two clusters
def NearestTwoSets(S1,S2):
    
    S1 = S1.T
    S2 = S2.T
    
    (l1,t) = S1.shape
    (l2,t) = S2.shape
    
    f = 0 
    
    if l1>= l2: 
        t = S1
        S1 = S2 
        S2 = t 
        f = 1
        
    tree = cKDTree(S2)
    
    id1 = 0 
    id2 = 0 
    dMin =  1e30 #some random very large number 
    
    for i in range(0,max(l1,l2)):
        
        d,j = tree.query( S1[i] , k=1)
        
        if d<= dMin: 
            id1 = i + 0 
            id2 = j + 0 
            dMin = d + 0 
            
    if f == 0: 
        return id1, id2, dMin
    else:
        return id2, id1, dMin
            
#This needs to be run to generate a connected graph for each cluster
#Must provide axis along which smoothing was done 
def OrderCluster(S,ax=0):
    
    In = np.argsort(S[ax], kind = 'heapsort')
    
    return In 
    
#This needs to connect each cluster with it's closest neighbor
def CreateConnectedGraph(S, clusters):
    
    k = max(clusters)
    
    d = {}
    
    for i in len(k):
        
        dMin = 1e30 
        id1_min = i
        id2_min = i
        closeCluster = i
        
        for j in len(k):
            
            if i == j:
                continue
            
            id1, id2, dist = NearestTwoSets(S[:][clusters==i],S[:][clusters==j])
            
            if dist <= dMin: 
                dMin = dist
                id1_min = id1
                id2_min = id2
                closeCluster = j 
                
        d[k] = (id1_min, id2_min, closeCluster)
    
    return d 
    
    
    
        
    
S1 = np.random.random((2,1000000))            
S2 = np.random.random((2,1000000)) 

#t1 = time()
#bla = NearestTwoSets(S1,S2)
#t2 = time()           
#print bla
#print (t2-t1)
        
t1 = time()
bla = OrderCluster(S1, 1)
t2 = time()    
print (t2-t1)    
    
        