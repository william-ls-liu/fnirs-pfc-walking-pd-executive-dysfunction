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
    start = events['Sample number'].iloc[0]
    end = events['Sample number'].iloc[1]
    for ch in list(data.columns):
        quiet_stance = data.loc[start:(end - 1), ch]
        quiet_stance_mean = math.fsum(quiet_stance) / len(quiet_stance)
        baseline_removed = data.loc[:, ch] - quiet_stance_mean
        corrected_df[ch] = baseline_removed
    
    return corrected_df
