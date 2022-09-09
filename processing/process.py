# Author: William Liu <liwi@ohsu.edu>

from .short_channel_correction import short_channel_correction
from .tddr import tddr
from .filter import fir_filter
from .baseline import baseline_subtraction
import pandas as pd
import numpy as np
import re
import os

def process_fnirs(data: dict, short_chs: list):
    """
    Helper method to run the processing algorithms.

    :param data: dictionary with raw data (a dataframe) and metadata (a dict)
    :param short_chs: list of the short (reference) channels
    :return: dataframe of processed fNIRS data
    """
    raw = data['data']
    metadata = data['metadata']
    sample_rate = int(float(metadata['Datafile sample rate']))
    short_data, long_data, events = _transform_data(raw, metadata, short_chs)
    short_channel_corrected = short_channel_correction(long_data, short_data)
    tddr_corrected = tddr(data=short_channel_corrected, sample_rate=sample_rate)
    filtered = fir_filter(data=tddr_corrected, fs=sample_rate)
    baseline = baseline_subtraction(data=filtered, events=events, frames_to_drop=sample_rate)
    # Add event column to processed dataframe
    baseline.insert(len(baseline.columns), 'Event', events['Event'])
    baseline.reset_index(inplace=True)
    baseline.rename(columns={'index': 'Sample number'}, inplace=True)

    return baseline


def _transform_data(df: pd.DataFrame, metadata: dict, short_chs: list) -> pd.DataFrame:
    """
    Separate raw data into separate DataFrames for the long and short channels. Return 
    a separate DataFrame with only the rows containing Event markers.
    """
    df = df.copy()
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
    events = df[df['Event'].notnull()]
    return_events = _verify_events(df, events, metadata)

    return short_data, long_data, return_events


def _verify_events(df: pd.DataFrame, events: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Check to make sure there are 3 events, otherwise artificially add events based on protocol.
    There should be 20s of quiet stance followed by 80s of walking for .mat files and 20s of quiet
    stance followed by 120s of walking for .txt files.

    :param events: dataframe containing rows where an event was marked
    :return: dataframe of events, with "artificial" events added if necessary
    """
    df = df.copy()
    fname, ftype = os.path.splitext(metadata['Export file'])
    fs = int(float(metadata['Datafile sample rate']))

    # Define length of segments (secs) based on protocol
    quiet = 20
    if ftype == '.txt':
        walk = 120
    else:
        walk = 80

    # If all 3 events were found, return unmodified dataframe
    if len(events) == 3:
        return events
    # If only two events were found, check to see which marker is
    # missing, then add an 'artificial'
    elif len(events) == 2:
        first = events['Sample number'].iloc[0]
        second = events['Sample number'].iloc[1]
        diff = second - first
        # If user forgot to input the last marker
        if diff < (30 * fs) and diff > (15 * fs):
            missing = second + (walk * fs)
            df.loc[missing, 'Event'] = 'Marker Added'
            events = df[df['Event'].notnull()]
            return events
        # If the user forgot to input the first marker
        elif diff < (130 * fs) and diff > (110 * fs):
            missing = first - (quiet * fs)
            df.loc[missing, 'Event'] = 'Marker Added'
            events = df[df['Event'].notnull()]
            return events
        # If the user forgot to input the middle marker
        else:
            missing = first + (quiet * fs)
            df.loc[missing, 'Event'] = 'Marker Added'
            events = df[df['Event'].notnull()]
            return events
    # If 1 event, or no events, were found, add in three events
    # based on protocol timing
    else:
        df['Event'] = np.nan
        first = 3 * fs  # First marker is 3 seconds into recording
        second = first + (quiet * fs)
        third = second + (walk * fs)
        df.loc[[first, second, third], 'Event'] = 'Marker Added'
        events = df[df['Event'].notnull()]
        return events
