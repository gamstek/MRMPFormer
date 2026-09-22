"""Mexican Hat (Ricker) 小波 与 连续小波变换（CWT）"""

import numpy as np


def ricker(points, a):
    """Mexican hat (Ricker) wavelet, 与旧版 scipy.signal.ricker 等价实现"""
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec ** 2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss


def cwt(data, wavelet, widths):
    """简化版连续小波变换，与旧版 scipy.signal.cwt 等价实现"""
    output = np.zeros((len(widths), len(data)))
    for ind, width in enumerate(widths):
        n = min(10 * width, len(data))
        wavelet_data = wavelet(int(n), width)
        output[ind] = np.convolve(data, wavelet_data, mode="same")
    return output
