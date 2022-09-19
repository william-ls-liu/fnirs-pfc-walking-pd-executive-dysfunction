# Author: William Liu <liwi@ohsu.edu>

import pandas as pd
import numpy as np
import scipy.signal as signal


def fir_filter(data: pd.DataFrame, order=1000, Wn=[0.01, 0.1],
               window='hamming', pass_zero='bandpass', fs=50) -> pd.DataFrame:
    taps = order + 1
    filt = signal.firwin(taps, Wn, window=window, pass_zero=pass_zero, fs=fs)

    # Apply the filter
    filtered_df = data.copy()
    for ch in list(filtered_df.columns):
        ch_asarray = np.array(filtered_df[ch], dtype='float64')
        filtered_df[ch] = signal.filtfilt(filt, [1], ch_asarray)

    return filtered_df
