import pandas as pd
import numpy as np
import math


def baseline_subtraction(data: pd.DataFrame, events: pd.DataFrame, frames_to_drop: int):
    """
    Subtract the initial quiet-stance phase of the recording.
    
    :param data: dataframe of processed fNIRS data
    :param events: dataframe of events that were tagged during fNIRS recording
    :return: dataframe of fNIRS data with baseline subtracted
    """
    if len(events) != 3:
        raise ValueError(f"The number of events found was {len(events)}, expected 3.")
    corrected_df = data.copy()
    # Subtract number of frames that were dropped initially to align with numpy indexing
    start = events['Sample number'].iloc[0] - frames_to_drop
    end = events['Sample number'].iloc[1] - frames_to_drop
    for ch in list(data.columns):
        ch_asarray = np.array(data[ch], dtype='float64')
        quiet_stance = ch_asarray[start:end]
        quiet_stance_mean = math.fsum(quiet_stance) / len(quiet_stance)
        ch_asarray -= quiet_stance_mean
        corrected_df[ch] = ch_asarray
    
    return corrected_df
