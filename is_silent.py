import numpy as np
from scipy import signal, stats

def is_silent(x):
    
    #scipy.signal.welch(x, fs=1.0, window='hanning', nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1
    
    f,psdw = signal.welch(x,window='hamming', nperseg=512)
    energy = 20*np.log10(np.sum(x**2))
    
    spectral_flatness = stats.gmean(psdw)/np.mean(psdw)
    
    threshold = 0.45
    silent = 0
    if np.bitwise_or(spectral_flatness >= threshold, energy < -50):
        silent = 1
        
    return silent, spectral_flatness