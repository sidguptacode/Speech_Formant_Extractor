import os
import math
import numpy as np
import scipy as sp
import IPython.display as ipd
import pandas as pd
import audioread
import json
import tensorflow.keras as keras
import librosa
import matplotlib.pyplot as plt
import matlab.engine
import pydub
import io

from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy import signal
from scipy.fft import fft, ifft
from scipy.signal import lfilter, hamming

from plt_signal import *

# Compute and plot the STFT of the voiceprint
# By default, signal creates it's own overlap
def compute_stft(clip, sample_rate, visualize=False):
    f, t, vp_stft = signal.stft(clip, fs=sample_rate)
    f = np.array(f).astype(int)
    vp_stft = np.abs(vp_stft)
    
    if visualize:
        plt.pcolormesh(t, f, np.abs(vp_stft))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
    return f, t, vp_stft

def find_peaks(f, t, vp_stft, stft_slice_index, visualize=False, figsize=None, lin_interp=True):
    vp_stft_T = np.transpose(vp_stft)

    num_times = len(t)
    num_freqs = len(f)
    max_freq = max(f)

    # For vp_stft_T_full, the rows indicate the time, and the columns indicate the frequencies.
    vp_stft_T_full = []
    vp_stft_T_full = np.zeros((num_times, max_freq + 1))

    for time in range(0, num_times):

        # For this time-slice, populate the frequencies that were recognized in the STFT. Everything else is 0.
        for i in range(0, num_freqs):
            freq = f[i]
            vp_stft_T_full[time][freq] = vp_stft_T[time][i]


    # For purposes of LPC, we will linearly interpolate all frequencies that were not detected.
    main_vp_stft_T_slice = vp_stft_T_full[stft_slice_index]
    freqs = f
    freq_magnitudes = [main_vp_stft_T_slice[freq] for freq in freqs]
    zero_freqs = [x for x in range(0, max_freq + 1)]
    if lin_interp:
        main_stft_slice_interp = np.interp(zero_freqs, f, freq_magnitudes)
    else:
        main_stft_slice_interp = main_vp_stft_T_slice # Without linear interpolation

    if visualize:
        plt.figure(figsize=figsize)
        plt.xlim(0, max_freq)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.plot(main_stft_slice_interp)
        # On top of the interpolated graph, plot the "real" frequencies in the form of a scatterplot.
        plt.scatter(x=freqs, y=freq_magnitudes)
        plt.show()

    main_vp_stft_slice = main_stft_slice_interp
    return main_vp_stft_slice

"""
    Librosa LPC Analysis
"""

def lpc_analysis(stft_slice, sample_rate, visualize=False):

    # Pick one slice of the STFT, and plot it as well as it's LPC
    lpc_coeffs = librosa.lpc(stft_slice, 10)
    vp_lpc = signal.lfilter([0] + -1*lpc_coeffs[1:], [1], stft_slice)

    if visualize:
        plt.figure(figsize=(20, 5))
        plt_signal(stft_slice, sample_rate, frame=None, player=False, xlabel="Frequency", ylabel="Magnitude")
        plt_signal(vp_lpc, sample_rate, player=False, frame=None,  color="red", xlabel="Frequency", ylabel="Magnitude")
        plt.show()

    rts = np.roots(lpc_coeffs)

    # Only choosing complex roots in the top plane
    rts = np.array([rt for rt in rts if np.imag(rt) > 0])

    angle_z = np.arctan2(np.imag(rts), np.real(rts))
    frqs = np.multiply((sample_rate / (2 *  np.pi)), angle_z)
    indices = np.argsort(frqs)
    frqs = frqs[indices]
    bw = np.multiply((-1/2)*(sample_rate / (2*np.pi)), (np.log(np.abs(rts[indices]))))

    formants = []
    a = 1
    for b in range(1, len(frqs)):
        if (frqs[b] > 100 and bw[b] < 500):
            formants.append(frqs[b])

    return formants
