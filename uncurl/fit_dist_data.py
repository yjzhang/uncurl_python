# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
import math as math



def GetDistFitError(Dat):
    #Assumes data to be in the form of a numpy matrix 
    # TODO: make this work for sparse inputs?
    # TODO: fixed number of bins, rather than use the maximum value?
    # use np.histogram
    Dat = np.round(Dat).astype(int)
    Dat2 = np.log(1 + Dat)
    BinDat = np.zeros(max(Dat)+1)
    Poiss = np.zeros(max(Dat)+1)
    Norm = np.zeros(max(Dat)+1)
    LogNorm = np.zeros(max(Dat)+1)

    m = np.mean(Dat)
    std = np.std(Dat, ddof=1)
    m_l = np.mean(Dat2)
    std_l = np.std(Dat2, ddof=1)

    #Create a bin of frequencies and generate frequencies based on distr
    for i in range(0,len(BinDat)):
        # this is EXTREMELY INEFFICIENT!!!!!
        # n^2 since Dat==i requires iterating through the whole array
        BinDat[i] = sum(Dat==i)
        Poiss[i] = poisson.pmf(i,m)
        Norm[i] = norm.pdf((i-m+1)/std)
        LogNorm[i] = norm.pdf((i-m_l)/std_l)
    BinDat = BinDat/sum(BinDat) + 0.0
    Poiss = Poiss/sum(Poiss) + 0.0
    Norm = Norm/sum(Norm) + 0.0
    LogNorm = LogNorm/sum(LogNorm) + 0.0
    #Get error for each distribution 
    PoissErr = np.linalg.norm(BinDat - Poiss)
    NormErr = np.linalg.norm(BinDat - Norm)
    LogNormErr = np.linalg.norm(BinDat - LogNorm)
    d = {}
    d['poiss'] = PoissErr
    d['norm'] = NormErr
    d['lognorm'] = LogNormErr
    return d

def NormPDF(x,mu,std):
    pi = math.pi
    temp = np.exp(-((x-mu)**2)/(2*std**2))/np.sqrt(2*pi*std**2)
    return temp

def DistFitDataset(Dat):
    """
    Given a data matrix, this returns the per-gene fit error for the
    Poisson, Normal, and Log-Normal distributions.

    Args:
        Dat (array): numpy array with shape (genes, cells)

    Returns:
        d (dict): 'poiss', 'norm', 'lognorm' give the fit error for each distribution.
    """
    #Assumes data to be in the form of a numpy matrix 
    (r,c) = Dat.shape
    Poiss = np.zeros(r)
    Norm = np.zeros(r)
    LogNorm = np.zeros(r)
    for i in range(r):
        temp = GetDistFitError(Dat[i])
        Poiss[i] = temp['poiss']
        Norm[i] = temp['norm']
        LogNorm[i] = temp['lognorm']
    d = {}
    d['poiss'] = Poiss
    d['norm'] = Norm
    d['lognorm'] = LogNorm
    return d


#Dat = np.array([[0,0,0,1,1,2,2,3,4],[0,0,0,1,1,1,3,5,7]])
#Dat = np.array([2,3,4,5])
#print GetDistFitError(Dat)
#n = 100
#Dat = np.random.poisson(lam = [[2]*n,[.5]*n], size = (2,n))
#d = DistFitDataset(Dat)    
