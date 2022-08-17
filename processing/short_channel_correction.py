import numpy as np
import pandas as pd
import re


def short_channel_correction(long_data: pd.DataFrame, short_data: pd.DataFrame):
    """
    Add docstring with sources for scholkmann 2014 and sager and berger 2005
    """
    corrected_df = long_data.copy()
    long_chs = list(long_data.columns)
    for long_ch in long_chs:
        short_ch = _find_short(long_ch, short_data)
        long_array = np.array(long_data[long_ch], dtype='float64')
        short_array = np.array(short_data[short_ch], dtype='float64')

        alpha = np.dot(short_array, long_array) / np.dot(short_array, short_array)
        corrected = long_array - (alpha * short_array)
        corrected_df[long_ch] = corrected

    return corrected_df


def _find_short(long_ch: str, short_data: pd.DataFrame):
    """Find and return the short channel for a given long channel."""
    short_chs = list(short_data.columns)
    receiver = long_ch.split('-')[0]
    oxygenation = long_ch.split(' ')[1]
    regex = f'{receiver}-Tx[0-9] {oxygenation}'
    for short_ch in short_chs:
        if re.match(regex, short_ch):
            return short_ch