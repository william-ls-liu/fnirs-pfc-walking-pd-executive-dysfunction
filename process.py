from parse_artinis_export import parse_artinis_export
import pandas as pd
import os
import re
import numpy as np
from scipy.signal import butter, sosfiltfilt
import math
from typing import Union

def process():
    pass

def _transform_data(df: pd.DataFrame) -> Union[np.array, pd.DataFrame]:
    """
    Create a numpy array of raw data from the DataFrame. Return numpy array without "Sample number" or "Event columns.
    Return a separate DataFrame with only the rows containing Event markers.
    """
    # DataFrame to numpy array
    df_asarray = df.iloc[:, 1:-1].to_numpy(dtype=np.float64)

    # DataFrame of events
    events = df[df['Event'] != '']

    return df_asarray, events

def _tddr(data: np.array) -> np.array:
    """
    Apply Temporal Derivative Distribution Repair algorithm.

    Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
    Temporal Derivative Distribution Repair (TDDR): A motion correction
    method for fNIRS. NeuroImage, 184, 171-179.
    https://doi.org/10.1016/j.neuroimage.2018.09.025
    """
    pass