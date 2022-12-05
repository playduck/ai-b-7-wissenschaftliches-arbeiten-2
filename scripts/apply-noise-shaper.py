import sys
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import filtfilt
import scipy as sp
import matplotlib.pyplot as plt

# generating a pure-tone signal
print("computing signal")
FS = 48000
frequency = 100
duration = 1
t = np.linspace(0, duration, FS * duration)

f0 = 1
f1 = 18000
k = np.power(f1 / f0, 1 / duration)
# signal = np.sin( 2 * np.pi * (f0 * np.power(k, t)) / duration)

signal = (np.sin(frequency * 2 * np.pi * t))

bits = 15 # 15 bits scale due to sign bit
scale = 2**(bits-1) # -1 for a bit headroom in anticipation of additive dither
signal_scaled = (scale * signal).astype(np.int16)

print("computing dither")
# TPDF dither using two uncorellated noise sources
dither = np.round((np.random.rand(len(signal)) + np.random.rand(len(signal))) / 2)
dither_scaled = (dither * (2**4)).astype(np.int16) # scale by 1 LSB

# apply dither
signal_scaled_dither = signal_scaled + dither_scaled

print("quantizing signals")
# quantization with right and left shifting
# sanity check
quantized_signal_shift = np.left_shift(np.right_shift(signal_scaled, 16-5), 16-5)
quantized_signal_dither_shift = np.left_shift(np.right_shift(signal_scaled_dither, 16-5), 16-5)

# quantization using a quantizer function
delta = (2**(16-5))
quantized_signal        = delta * np.floor((signal_scaled        / delta) + (1/2))
quantized_signal_dither = delta * np.floor((signal_scaled_dither / delta) + (1/2))

# singal error
error = quantized_signal_dither - signal_scaled

print("loading filter coefficients")
# get filter
kernel = np.array(pd.read_csv("filter.csv")["0"])

print("applying noise shaping")
# apply noise shaping
quantized_signal_noise_shaped = [0] * len(signal)
filter_buffer = [0] * len(kernel)
filtered_error = 0

for n in range(len(signal)):
    if n % (FS//10) == 0:
        print(n/len(signal))

    sample = signal_scaled[n] + filtered_error
    dithered_sample = sample + dither[n]
    quantized_sample = delta * np.floor((dithered_sample / delta) + (1/2))

    error = quantized_sample - sample

    filter_buffer.append(error)

    filtered_error = 0
    # if n < len(kernel):
        # filtered_error = 0
    # else:
    for k in range(len(kernel)):
        filtered_error = filtered_error + filter_buffer[n-k] * kernel[k]

    quantized_signal_noise_shaped[n] = (sample)


print("displaying figure")

fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)

ax[0].plot(np.abs(fft(signal_scaled)[1:FS//2]))
ax[0].loglog()
ax[0].set_ylim(100, 5_000_000_000)
# ax[0].plot(signal_scaled)
ylim=ax[0].get_ylim()

ax[1].plot(np.abs(fft(quantized_signal)[1:FS//2]))
ax[1].loglog()
# ax[1].plot(quantized_signal)
ax[1].set_ylim(ylim)

ax[2].plot(np.abs(fft(quantized_signal_dither)[1:FS//2]))
ax[2].loglog()
# ax[2].plot(quantized_signal_dither)
ax[2].set_ylim(ylim)

ax[3].plot(np.abs(fft(quantized_signal_noise_shaped)[1:FS//2]))
ax[3].loglog()
# ax[3].plot(quantized_signal_noise_shaped)
ax[3].set_ylim(ylim)
ax[3].set_xlabel("f in Hz")

plt.show()
