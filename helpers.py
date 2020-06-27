import numpy as np

def parabolic_interpolation(a,b,c):
    # if a - 2*b + c == 0:
    #     pos = 0
    # else:
    #     pos = 0.5 * ((a-c)/(a - 2*b + c))
    
    pos = 0.5 * ((a-c)/(a - 2*b + c))
    peak = b - 0.25*(a-c)*pos
    return peak,pos

def nextpow2(x):
    return 1 if x == 0 else np.ceil(np.log2(x))

def poisson_window(M,alpha):
    return np.exp(-0.5*alpha*np.arange(0,M)/(M-1))