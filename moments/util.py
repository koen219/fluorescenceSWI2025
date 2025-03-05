# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:12:00 2025

@author: joriswinden
"""
import numpy as np

def order_params(x):
    (t1,t2,p) = x
    if t1 < t2:
        return (t2,t1,1-p)
    else:
        return x
    
def counts_to_times(counts,times):
    return np.concatenate([np.full(n,times[i]) for (i,n) in enumerate(counts.astype(int))])

def mom_est(data):
    m1 = np.mean(data)
    m2 = np.mean(data**2)
    m3 = np.mean(data**3)
    return params_from_moments(m1,m2,m3)    

def params_from_moments(m1,m2,m3):
    a = 6 * (2*(m1**2) - m2)
    b = 2 * (m3 - 3 * m1*m2)
    c = 3*(m2**2) - 2*m1*m3

    D = max(0,b**2 - 4*a*c) # set to zero if negative
    if D > 0:
        t1 = (-b + np.sqrt(D)) / (2 * a)
        t2 = (-b - np.sqrt(D)) / (2 * a)
        p = (m1 - t1) / (t2 - t1)
        
        t1 = max(0,t1)
        t2 = max(0,t2)
        p = min(1,max(0,p))
    else:
        t1 = m1
        t2 = m1
        p = 0.5
    return (t1,t2,p) 


def mle_est(data,realt1=None,realt2=None,M=20,eps=1e-2):
    # 'reasonable' starting values
    t1 = np.mean(data) * 2
    t2 = t1 / 4
    p = 0.5
    
    for i in range(M):
        (oldt1,oldt2,oldp) = (t1,t2,p)
        (t1,t2,p) = mle_em_update_params(data,t1,t2,p)
        
        # possibly replace times with 'true' values
        t1 = t1 if realt1 is None else realt1
        t2 = t2 if realt2 is None else realt2
        
        if abs(oldt1-t1) < eps and abs(oldt2 - t2) < eps and abs(oldp - p) < eps:
            break
    return order_params((t1,t2,p))

def mle_em_update_params(data,t1,t2,p):
    pleft = calc_p_left(data,t1,t2,p)
    pright = 1 - pleft
    newt1 = np.sum(pleft * data) / np.sum(pleft)
    newt2 = np.sum(pright * data) / np.sum(pright)
    newx = np.sum(pleft) / np.sum(pright)
    newp = newx / (1 + newx)
    return (newt1,newt2,newp)

def calc_p_left(data,t1,t2,p):
    num = p * 1/t1 * np.exp(-data / t1)
    denom = p * 1/t1 * np.exp(-data / t1) + (1-p)* 1/t2 * np.exp(-data / t2)
    return np.divide(num,denom)