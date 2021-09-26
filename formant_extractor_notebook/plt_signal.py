import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

# Helper function used to plot signals (and play them)
def plt_signal(x, sr, frame=None, player=False, color=None, xlabel='Time (samples)', ylabel='Amplitude'):
    if frame is None:
        frame = [0, x.shape[0]]
    plt.plot(x, color=color)
    plt.xlim(frame)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if player:
        ipd.display(ipd.Audio(data=x, rate=sr))
