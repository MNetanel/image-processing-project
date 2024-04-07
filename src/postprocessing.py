'''
Implementation of various postprocessing methods for processing rPPG signals.
SOURCE: rPPG-Toolbox https://github.com/ubicomplab/rPPG-Toolbox/
See LICENSE.txt for license.
'''

import numpy as np
import scipy
from scipy import sparse
from scipy.signal import butter
from typing import Literal

def _detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(
        ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs=30):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def process_signal(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5):
    processed_signal = _detrend(ppg_signal, 100)
    [b, a] = butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
    processed_signal = scipy.signal.filtfilt(b, a, np.double(processed_signal))
    return processed_signal


def ppg2hr(ppg_signal, fs=30, do_process=True,
           method: Literal['fft', 'peaks'] = 'fft',
           low_pass=0.75, high_pass=2.5):
    if do_process:
        processed_signal = process_signal(ppg_signal, fs, low_pass, high_pass)
    if method == 'fft':
        return _calculate_fft_hr(ppg_signal, fs, low_pass, high_pass)
    if method == 'peaks':
        return _calculate_peak_hr(ppg_signal, fs)
    
def ppg2hr_by_window(ppg_signal, window_size_seconds=10, fs=30,
                     do_process=True, method: Literal['fft', 'peaks'] = 'fft',
                     low_pass=0.75, high_pass=2.5):
    hrs = []
    window_size_frames = window_size_seconds * fs
    window_size_frames = min(window_size_frames, len(ppg_signal))
    for i in range(0, len(ppg_signal), window_size_frames):
        bvp_window = ppg_signal[i:i + window_size_frames]
        if len(bvp_window) < 9:
            print(f"Window frame size of {len(bvp_window)} is smaller than minimum pad length of 9. Window ignored!")
            continue
        hr = ppg2hr(bvp_window, fs=fs, do_process=do_process,
                    method=method, low_pass=low_pass, high_pass=high_pass)
        hrs.append(hr)
    return np.array(hrs)