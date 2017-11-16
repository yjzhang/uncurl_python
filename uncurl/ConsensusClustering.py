# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 14:20:01 2017

@author: Sumit
"""

import numpy as np 
from scipy import sparse

def Cluster2Adj(Cluster):
    
    M = len(Cluster)
           
    Arr = np.zeros((M,M)) 
    
    for i in range(M):
        
        for j in range(i,M):
            
            if Cluster[i] == Cluster[j]:
                Arr[i,j] = 1.0
                Arr[j,i] = 1.0
                   
               
    return Arr 


def GetClusterDistance(Cluster1, Cluster2):
    
    C1 = Cluster2Adj(Cluster1)
    C2 = Cluster2Adj(Cluster2)
    
    d = np.sum(np.abs(C1-C2))
    
    return d


def PickBestCluster(InstanceList):
    
    #print InstanceList
    
    BestClustering = InstanceList[0]
    
    dBest = 1.0e20
    
    l = len(InstanceList)
    
    for i in range(l):
        
        d = 0.0  
        
        for j in range(l):
            
            if i == j:
                continue
            
            C_i = InstanceList[i]
            C_j = InstanceList[j]
            
            
            dt = GetClusterDistance(C_i,C_j)
            
            d += dt + 0.0 
            
        if d <= dBest: 
            
            BestClustering = C_i 
            dBest = d + 0.0 
            
    return BestClustering
    
    
    
    

           