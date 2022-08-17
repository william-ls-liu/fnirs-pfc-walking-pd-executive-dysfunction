import pandas as pd
import numpy as np
import scipy.signal as signal


def fir_filter(data: pd.DataFrame, order=1000, Wn=[0.01, 0.1], 
               window='hamming', pass_zero='bandpass', fs=50) -> pd.DataFrame:
    print(f"""
    Building FIR Filter with following parameters
    ---------------------------------------------
    Order: {order}
    Cutoff frequency(ies): {Wn}
    Window: {window}
    Type: {pass_zero}
    Sample rate: {fs}
    """)

    taps = order + 1
    filt = signal.firwin(taps, Wn, pass_zero=pass_zero, fs=fs)

    # Apply the filter
    filtered_df = data.copy()
    for ch in list(data.columns):
        ch_asarray = np.array(data[ch], dtype='float64')
        filtered_df[ch] = signal.filtfilt(filt, [1])
    
    return filtered_df