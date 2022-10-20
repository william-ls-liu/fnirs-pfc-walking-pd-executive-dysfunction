# Author: William Liu <liwi@ohsu.edu>

import pandas as pd


def create_segments(df: pd.DataFrame) -> dict:
    """
    Split trial into segments based on 'Event' markers.

    :param df: dataframe of processed fnirs
    :return: dict with keys as individual segments and values
             as dataframes with fnirs data for given segment
    """
    df_copy = df.copy()
    if type(df_copy) != pd.DataFrame:
        raise TypeError(f"Must provide dataframe, not {type(df_copy)}.")

    events = df_copy[df_copy['Event'].notnull()]
    if len(events) != 3:
        raise IndexError(f"Expected 3 event markers, found {len(events)}.")

    # Define a dictionary to store dataframe for each segment
    segments = dict()

    # Use event markers to define 2 segments.
    # One is quiet stance, other is walking.
    for idx, seg in enumerate(['Quiet Stance', 'Walking']):
        start = events.index[idx]
        end = events.index[idx + 1]
        segments[seg] = df_copy.iloc[start:end]
    # Create early and late phase segments during walking
    start = events.index[1]
    end = events.index[2]
    gap = end - start
    mid = gap // 2
    mid += start
    early = df_copy.iloc[start:mid]
    late = df_copy.iloc[mid:end]
    # Add early and late to segments
    segments['early'] = early
    segments['late'] = late

    return segments
