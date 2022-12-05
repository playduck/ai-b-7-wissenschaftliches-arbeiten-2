#!/usr/bin/env python3

import sys
import numpy as np
import scipy as sp
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.fft import ifft
from scipy.fft import fft
from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt

def ISO226(phon):
    f = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                  800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])
    af = np.array([0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301, 0.288, 0.276,
                   0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301])
    Lu = np.array([-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1, -
                   0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1])
    Tf = np.array([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2,
                   4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3])

    if((phon < 0) | (phon > 90)):
        return Lp, f
    else:
        Ln = phon

        # SPL from loudness level (iso226 sect 4.1)
        Af = 4.47E-3 * (pow(10, (0.025*Ln)) - 1.15) + \
            pow((0.4 * pow(10, (((Tf+Lu)/10)-9))), af)

        Lp = ((10./af) * np.log10(Af)) - Lu + 94

        return Lp, f

FS = 48000
LINE_SAMPLES = 1024

# Get frequency response of absolute threshold of human hearing
spl, f = ISO226(0)

# resample and interpolate curve as spline
newF = np.linspace(0, FS//2, LINE_SAMPLES)
spline = interp1d(f, spl, kind="cubic", fill_value="extrapolate", bounds_error=False)

# soft-clamp spline to maximum excursion
spline_interp = spline(newF)

MAX_EXCURSION_HIGH = 20
MAX_EXCURSION_MAX = 0.5
spline_clamped = np.tanh((spline_interp/MAX_EXCURSION_HIGH)) * MAX_EXCURSION_HIGH



# hard-clamp
# spline_clamped = np.clip(spline_clamped, -MAX_EXCURSION_MAX, MAX_EXCURSION_MAX)
spline_clamped = (spline_clamped / np.max(np.abs(spline_clamped)))
# spline_clamped = 1 - spline_clamped

def avg(g, *args):
    # absolute of numeric integral
    # return np.abs( (1/len(args[0])) * np.sum(args[0] + g))
    return np.power( (1/len(args[0])) * np.sum(args[0] + g) , 2)

# optimize spline gain (y offset) to achieve equal amplification and attenuation
# sum of all (shaped) noise must be 0 or greater (not negative, since energy musnt be lost)
optimized_gain_average = sp.optimize.minimize(
    fun=avg, x0=[0], args=(spline_clamped), method='Powell', tol=1e-16, options={'disp': False})

# apply optimized gain
OPTIMIZED_GAIN = optimized_gain_average.x[0]
optimized_spline = (spline_clamped + OPTIMIZED_GAIN)

# interpolate and evaluate
spline_zero = np.append(optimized_spline[0], optimized_spline)
omega = np.linspace(0, np.pi, len(spline_zero))
Hd = interp1d(omega, spline_zero, kind="cubic", fill_value="extrapolate", bounds_error=False)
N = 512
k = np.linspace(0, N-1, N)
Hk = Hd( (2 * np.pi * k )/N )

if N % 2 == 0:
    UL = (N//2) - 1 # even
else:
    UL = (N-1) // 2 # odd

# perform inverse fft and scale
h = np.real(ifft(Hk))
h = h / np.max(np.abs(h))
h = h * (2 / (np.pi))

# shift fft to generate FIR kernel
left = h[UL:]
right = h[:UL]
kernel = np.append(left, right)
TAPS = len(kernel)
response = kernel

# window kernel
window = np.cos(np.linspace(-np.pi, np.pi, TAPS)) / 2 + 0.5
response = response * window

filter_x = np.linspace(0, TAPS + 1, TAPS)

# save kernel
response_df = pd.DataFrame(response)
response_df.to_csv("filter.csv")

# simulate response
w, h = signal.freqz(b=response[TAPS//2 + 1:], worN=LINE_SAMPLES, fs=FS)
h = h / np.abs(np.max(h))
h = h + OPTIMIZED_GAIN

# calculate error
abs_error = np.abs(np.real(optimized_spline) - h.real)
rel_error = abs_error / np.abs(np.real(optimized_spline))

duration = 2
signal_time = np.linspace(0, duration, FS * duration)
f0 = 1
f1 = 18000
k = np.power(f1 / f0, 1 / duration)
chirp = (np.sin( 2 * np.pi * (f0 * np.power(k, signal_time)) / duration) * 2**15 -1)

wavfile.write("chirp-normal.wav", FS, chirp)

filtered = signal.filtfilt(response, 1, chirp)
wavfile.write("chirp-filter.wav", FS, filtered)

# plot figures
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.3, wspace=0.3)

ax = fig.add_subplot(221)
ax.set_xscale("log", base=10)
ax.set_title("Gain optimized ATH for ISO226")
ax.set_xlabel("f")
ax.set_ylabel("dB")
ax.plot([newF.min(), newF.max()], [0, 0])
ax.plot(newF, optimized_spline, lw=2)

ax2 = fig.add_subplot(222)
ax2.plot([0, TAPS], [0, 0])
ax2.set_title("FIR filter taps")
ax2.set_xlabel("tap")
ax2.set_ylabel("value")
ax2.stem(filter_x, response)
ax2.plot(filter_x, window, lw=2)

ax3 = fig.add_subplot(223)
ax3.set_xscale("log", base=10)
ax3.set_title("Normalized ATH vs simulated FIR")
ax3.set_xlabel("f")
ax3.set_ylabel("")
ax3.plot([newF.min(), newF.max()], [0, 0])
ax3.plot(newF, optimized_spline)
ax3.plot(w, h.real)
# ax3.plot(np.real(fft(response)))
# h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
# ax3.plot(w, h_Phase, lw=2)

df1 = pd.DataFrame({"frequency": newF, "amplitude":optimized_spline})
df2 = pd.DataFrame({"frequency": w, "amplitude":h.real})
df1.to_csv("ath.csv")
df2.to_csv("fir.csv")

ax4 = fig.add_subplot(224)
ax4.set_xscale("log", base=10)
ax4.set_title("Error between ATH and FIR")
ax4.set_xlabel("f")
ax4.set_ylabel("Delta")
ax4.plot([newF.min(), newF.max()], [0, 0])
ax4.plot(newF, abs_error, lw=2)
ax4.plot(newF, rel_error, lw=1)
max_err = np.max(abs_error)
ax4.set_ylim(-1, max_err * 2)

plt.show()
