from .short_channel_correction import short_channel_correction
from .tddr import tddr
from .filter import fir_filter
from .baseline import baseline_subtraction
import pandas as pd
import re


def process(data: dict, short_chs: list):
    """
    Helper method to run the processing algorithms.

    :param data: dictionary with raw data (a dataframe) and metadata (a dict)
    :param short_chs: list of the short (reference) channels
    :return: dataframe of processed fNIRS data
    """
    raw = data['data']
    metadata = data['metadata']
    sample_rate = int(float(metadata['Datafile sample rate:']))
    short_data, long_data, events = _transform_data(raw, short_chs)
    short_channel_corrected = short_channel_correction(long_data, short_data)
    tddr_corrected = tddr(data=short_channel_corrected, sample_rate=sample_rate)
    filtered = fir_filter(data=tddr_corrected, fs=sample_rate)
    baseline = baseline_subtraction(data=filtered, events=events, frames_to_drop=sample_rate)

    return baseline, filtered, events


def _transform_data(df: pd.DataFrame, short_chs: list) -> pd.DataFrame:
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
