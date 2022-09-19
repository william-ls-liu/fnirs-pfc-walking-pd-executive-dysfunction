# Author: William Liu <liwi@ohsu.edu>

import numpy as np
import pandas as pd
import re


def ssc_correction(long_data: pd.DataFrame, short_data: pd.DataFrame):
    """
    Apply short channel correction technique to remove the superficial
    component of the probed tissue.

    Felix Scholkmann et al 2014 Physiol. Meas. 35 717

    :param long_data: dataframe containing fNIRS data for the long channels
    :param short_data: dataframe containing fNIRS data for the short reference
                       channels
    :return: dataframe of corrected long channels
    """
    short_data_copy = short_data.copy()
    long_data_copy = long_data.copy()
    corrected_df = long_data.copy()
    long_chs = list(long_data.columns)
    for long_ch in long_chs:
        short_ch = _find_short(long_ch, short_data_copy)
        long_array = np.array(long_data_copy[long_ch], dtype='float64')
        short_array = np.array(short_data_copy[short_ch], dtype='float64')

        alpha = (
            np.dot(short_array, long_array) / np.dot(short_array, short_array)
            )
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

    raise KeyError(f"Could not find matching short channel for {long_ch}")
