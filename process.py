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

def _transform_data(df: pd.DataFrame, short_chs: list) -> Union[np.array, pd.DataFrame]:
    """
    Separate raw data into separate DataFrames for the long and short channels. Return a separate DataFrame with only 
    the rows containing Event markers.
    """
    # Get DataFrame with only the short channels
    short_regex = '|'.join(short_chs)
    short_data = df.filter(regex=short_regex)

    # Get DataFrame with only the long channels
    all_chs = list(df.columns)
    long_chs = list()
    for ch in all_chs:
        if not ('Sample number' in ch or 'Event' in ch):
            if not re.match(short_regex, ch):
                long_chs.append(ch)
    long_regex = '|'.join(long_chs)
    long_data = df.filter(regex=long_regex)

    # DataFrame of events
    events = df[df['Event'] != '']

    return short_data, long_data, events
