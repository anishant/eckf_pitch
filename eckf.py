import numpy as np
from scipy import signal,stats

from helpers import parabolic_interpolation
from is_silent import is_silent
from harmonic_change import harmonic_change_detector

def eckf_pitch_modified(y, fs, blockSize, c, numBufToWait, npeaks, nsemitones=2):
    
#y - incoming noisy signal (row vector) 
#fs - sampling rate
#f0 - estimated pitch
#c - parameter for determining process noise
#numBufToWait - number of buffers to wait after note onset - this will vary
#from instrument to instrument, depending on how strong its attack is
#nsemitones - number of semitones of pitch change to occur for a new note
#to be detected
#amp - estimated amplitude of fundamental
#phi - estimated phase of fundamental
#x_est - estimated fundamental component of signal


    nframes = int(np.ceil((y.size)/blockSize))
    padding = np.zeros((nframes*blockSize - y.size))
    
    y = np.concatenate((y, padding))
    
    
    blocks = 1
    onset_pos = []

    Ts = 1/fs
    H =[0,0.5,0.5]
    H = np.array(H).reshape(len(H), 1)

    K = np.zeros((3,1))
    flag = -1
    Kthres = 0.01
    x_est = np.zeros(len(y))
    amp = np.zeros(len(y))
    phase = np.zeros(len(y))
    Q = np.zeros(len(y))
    f0 = np.zeros(len(y))


    P_track = np.zeros(len(y))
    K_track = np.zeros(len(y))
    n = 0
    start_pos = 0
#     %state of current frame - initially silent
    silent_cur = 1
    harm_prev = 0
    spf = []
    
    hcd = harmonic_change_detector()
    

    while(blocks <= nframes):

        end_pos = start_pos + blockSize - 1
        if(end_pos >= max(np.shape(y))):
            break
        y_frame = y[start_pos:start_pos + blockSize]
        

        silent_prev = silent_cur
        silent_cur, cur_spf = is_silent(y_frame)

        spf = np.matrix([[spf],[cur_spf]])
    #     %if current frame is silent, then continue
        if(silent_cur == 1):
            flag = 0
            start_pos = start_pos+blockSize
            n = start_pos
            continue
        else:

            if (start_pos > blockSize-1):
               y_prev = y[start_pos-blockSize:start_pos]
            else:
                y_prev = []

            harm_cur,_,_,_ = hcd(y_prev,y_frame,fs,npeaks,nsemitones)


            if ((silent_prev == 1 and silent_cur == 0) or (harm_prev == 0 and harm_cur == 1)):
                onset_pos = [onset_pos, [start_pos]]
               # onset_pos = [onset_pos, start_pos]
                count = 0

                while(count < numBufToWait):
                    count = count+1
                    start_pos = start_pos+blockSize

                if(start_pos + blockSize  < y.shape[0]):
                    y_frame = y[start_pos:start_pos+blockSize]
                    if(n>0):
                        f0[n:start_pos] = f0[n-1]
                        amp[n:start_pos] = amp[n-1]
                        phase[n:start_pos] = phase[n-1]
                    flag = 1
                    n = start_pos
                else:
                    break

        if(flag == 1):
            if (start_pos > blockSize-1):
                y_prev = y[start_pos-blockSize:start_pos]
            else:
                y_prev = []
                
            harm_cur, f1, a1, phi1 = hcd(y_prev.T,y_frame.T,fs,npeaks,nsemitones)
            # print('{}   {}  {}'.format(f1,a1,phi1))
            x0 = [0+0j,0+0j,0+0j]
            x0[0] = np.exp(1j*2*np.pi*f1*Ts)
            x0[1] = a1*np.exp(1j*2*np.pi*f1*n*Ts + 1j*phi1)
            x0[2] = a1*np.exp(-1j*2*np.pi*f1*n*Ts - 1j*phi1)
            x0 = np.array(x0).reshape(3, 1)
            

            P0 = np.zeros((3,3))
            
            if(abs(min(K)) < Kthres):
                # print('Covariance matrix reset at time {} seconds'.format((n-1)/fs))
                P_last = P0
                x_last = x0
                flag = 0
                start_pos = start_pos - count*blockSize
                n = start_pos

        k = 0
        while (k < blockSize-1):
            K = (np.dot(P_last, H))/(np.dot(H.reshape(1, len(H)), np.dot(P_last, H).reshape(3, 1))[0][0] + 1)
            P = P_last - np.matmul(np.dot(K, H.reshape(1, len(H))), P_last)
            x = x_last + K*(y_frame[k] - np.dot(H.reshape(1, len(H)), x_last))
            x_next = np.array([x[0][0],x[0][0]*x[1][0],x[2][0]/x[0][0]]).reshape(3, 1)
            F = [[1,0,0],[x[1][0],x[0][0],0],[-x[2][0]/(x[0][0]**2),0,1/x[0][0]]]
            Q[n] = 10**-(c-np.abs(y_frame[k] - np.dot(H.reshape(1, len(H)), x)[0][0]))
            P_next = F*P*np.array(F).T + Q[n]*np.eye(3)
            
            
            # with open('ppospy.txt', 'a') as f:
            #         f.write('{}+{}i\n'.format(x[0][0].real,x[0][0].imag))

            ####### WRONG VALUE to temp
            temp = np.log(x[0][0])/(1j*Ts*2*np.pi)
            f0[n] = np.abs(temp)
                    

            amp[n] = np.abs(x[1])
            phase[n] = np.abs(-1j * (np.log(x[1]/amp[n])-(2*np.pi*f0[n]*Ts*n)))
           # x_est[n] = np.dot(H.reshape(1, len(H)), x)[0][0]
            x_est[n] = np.abs(np.vdot(H,x))
            P_last = P_next
            x_last = x_next
            P_track[n] = np.linalg.norm(P)
            K_track[n] = np.linalg.norm(K)
            n = n + 1
            k = k + 1

        start_pos = start_pos + blockSize
        harm_prev = harm_cur


    return f0,amp,phase,x_est,onset_pos


