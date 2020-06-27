import numpy as np
from scipy import signal, stats
from helpers import poisson_window, parabolic_interpolation

from scipy.fftpack import fftshift
from scipy.fftpack import fft


class harmonic_change_detector(object):
    def __init__(self):
        self.minf = 0
        self.nfft = 0
        self.fbins = []
        self.win = []
        self.nbins_below50 = 0
        self.exp_win = []

    def __call__(self,xprev,xcur,fs,npeaks,nsemitones):
        if len(self.fbins) == 0:
            self.minf = max(xcur.shape)
            self.nfft = int(pow(2, np.ceil(np.log2(4*(self.minf+1)))))
            self.fbins = np.linspace(-fs/2,fs/2,self.nfft)
            self.win = signal.windows.blackman(self.minf)
            self.nbins_below50 = int(round(50/(fs/2*self.nfft)))
            self.fbins = self.fbins[int(self.nfft/2)-1 + self.nbins_below50:]
            self.exp_win = poisson_window(self.fbins.shape[0], 5)
        
        xcur = (xcur - np.mean(xcur))* self.win
        X = fftshift(fft(xcur, self.nfft))
        X = X[(int(self.nfft/2)-1) + self.nbins_below50:] 
        mag_cur = (np.absolute(X)/np.mean(self.win))* self.exp_win

    
        peak_indices, props = signal.find_peaks(mag_cur)
        pval_cur = mag_cur[peak_indices]
        ppos_cur = peak_indices
        ind = pval_cur.argsort()[::-1][:npeaks]
        ind.sort()
        # Gets indices of npeak biggest peaks, 
        ppos_cur_sort = ppos_cur[ind]
        fpeaks_cur = self.fbins[ppos_cur_sort]
        
    
        phi = np.angle(X)
        mpos = ppos_cur_sort[0]
        amp,pos = parabolic_interpolation(mag_cur[mpos-1],mag_cur[mpos],mag_cur[mpos+1])
        amp = amp/self.nfft
        f0_est, _ = stats.mode(np.round(np.diff(fpeaks_cur)))
        phase,pos = parabolic_interpolation(phi[mpos-1],phi[mpos],phi[mpos+1])


        if len(xprev) != 0:
            xprev = (xprev-np.mean(xprev))*self.win
            X_prev = fftshift(fft(xprev, self.nfft))
            X_prev = X_prev[int(self.nfft/2)-1 + self.nbins_below50:] 
            mag_prev = (np.abs(X_prev)/np.mean(self.win)) * self.exp_win

            peak_indices, _ = signal.find_peaks(mag_prev)
            pval_prev = mag_prev[peak_indices]
            ppos_prev = peak_indices
            #ind = np.argsort(pval_prev)
            ind = pval_prev.argsort()[::-1][:npeaks]
            ind.sort()
            #ind = ind[::-1]
            #pval_prev_sort= pval_prev[ind] #Descending order of peaks
            #ind = np.sort(ind[:npeaks])
        # pval_prev_sort = pval_prev[ind]
            ppos_prev_sort = ppos_prev[ind]
            fpeaks_prev = self.fbins[ppos_prev_sort]
            # with open('fprevpy.txt', 'a+') as f:
            #         f.write("{}".format(fpeaks_prev))

            
    ###############################################################################        
            # binedges = np.arange(-50, 50)+np.abs(np.diff(fpeaks_cur) - np.diff(fpeaks_prev))
            # ix = 50
    ###############################################################################
            n,binedges = np.histogram(np.abs(np.diff(fpeaks_cur) - np.diff(fpeaks_prev)),100)
            c = np.diff(binedges)/2
            ix = np.nanargmax(n)
            mx = n[ix]
            cent_dev = 1200*np.log2(f0_est/(f0_est+c[ix]))
            
            if(np.abs(cent_dev) >= nsemitones*100):
                flag_cur = 1
            else:
                flag_cur = 0
        else:
            flag_cur = 0
        
        
        return flag_cur,f0_est,amp,phase