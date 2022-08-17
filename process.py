from short_channel_correction import short_channel_correction
from tddr import tddr
import pandas as pd
import re


def process(data: dict, short_chs: list):
    raw = data['data']
    metadata = data['metadata']
    sample_rate = int(float(metadata['Datafile sample rate:']))
    short_data, long_data, events = _transform_data(raw, short_chs)
    short_channel_corrected = short_channel_correction(long_data, short_data)
    tddr_corrected = tddr(data=short_channel_corrected, sample_rate=sample_rate)

    return tddr_corrected


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
