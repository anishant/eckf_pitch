import numpy as np
import librosa as lr
from eckf import eckf_pitch_modified

# snd,fs = lr.load('SrGmPdNS^.wav',44100)
snd,fs = lr.load('SRG.wav',44100)
time = np.linspace(0, len(snd) / fs, num=len(snd))
# from scipy.io import wavfile as wv
# fs, s = wv.read('SRG.wav')
# if s.dtype == 'int16':
#     nb_bits = 16 # -> 16-bit wav files
# elif s.dtype == 'int32':
#     nb_bits = 32 # -> 32-bit wav files
# max_nb_bit = float(2 ** (nb_bits - 1))
# s = s / (max_nb_bit + 1.0)

#print(np.max(s))
[f0_est,amp,phi,x_est,onset_pos] = eckf_pitch_modified(snd, fs, 2048, 9, 2, 6, 0.25)

with open('check.txt', 'w') as f:
    for value in f0_est:
        f.write("%f," % value)