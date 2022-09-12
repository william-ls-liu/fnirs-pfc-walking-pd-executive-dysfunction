# Author: William Liu <liwi@ohsu.edu>

import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import numpy as np
import math

def read_mat(file_path: str) -> dict:
    """
    Read Artinis export of raw fNIRS data in the .mat format.

    :param file_path: path to the raw data file
    :return: dictionary of metadata and raw fnirs data
    """
    # Load the .mat file into a dictionary
    mat_dict = sio.loadmat(file_path)

    # Get metadata
    fs = mat_dict['nirs_data']['Fs'][0, 0][0, 0]
    metadata = {'Datafile sample rate': fs, 'Export file': file_path}

    # Get the channel labels. 'labels' is an array of arrays
    # so need to unpack into a list.
    labels = mat_dict['nirs_data']['label'][0, 0][0]
    labels_list = list()
    for label in labels:
        labels_list.append(label[0])

    # Get oxy and dxy data (type is numpy.ndarray)
    oxyvals = mat_dict['nirs_data']['oxyvals'][0, 0]
    dxyvals = mat_dict['nirs_data']['dxyvals'][0, 0]

    # Create dataframe for oxy and dxy, then merge
    oxy = pd.DataFrame(data=oxyvals, columns=[s + ' O2Hb' for s in labels_list])
    dxy = pd.DataFrame(data=dxyvals, columns=[s + ' HHb' for s in labels_list])
    df = pd.concat([oxy, dxy], axis=1)

    # Get event markers, add to dataframe
    events = _get_events(mat_dict)
    df.insert(len(df.columns), 'Event', events)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Sample number'}, inplace=True)

    # Drop initial ~1s of recording
    df.drop(df.index[range(fs)], inplace=True)

    # Scale values
    scaled_df = _scale_values(df)
    
    return {'metadata': metadata, 'data': scaled_df}


def _get_events(raw: dict) -> pd.Series:
    """
    Extract event markers from PortaSync signal.

    Events are denoted by 'high' values in the signal on channel 2 in ADvalues.
    By finding the peaks in the signal we can get the frame where an event was
    marked by the person collecting data.

    :param raw: a dictionary of loaded .mat data
    :return: a pd.Series containing the frames where events were marked
    """
    # If column containing event signal is not present, return series that will
    # make entire 'Event' column np.nan
    if raw['nirs_data']['ADvalues'][0, 0].shape[1] != 3:
        events = pd.Series(data=[np.nan])
        return events

    raw_event_signal = raw['nirs_data']['ADvalues'][0, 0][:, 1]
    peaks, _ = signal.find_peaks(raw_event_signal, height=0.02)
    markers = ['S1', 'W1', 'S2']
    used_markers = list()
    for i, _ in enumerate(peaks):
        used_markers.append(markers[i])
    events = pd.Series(data=used_markers, index=peaks)

    return events


def _scale_values(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the output by subtracing the mean of initial 15 frames."""
    scaled = df.copy()
    for ch in list(scaled.columns):
        if 'Sample number' in ch or 'Event' in ch:
            continue
        ch_asarray = np.array(df[ch], dtype=np.float64)
        initial_frames = ch_asarray[:15]
        mean = math.fsum(initial_frames) / len(initial_frames)
        ch_asarray -= mean
        scaled[ch] = ch_asarray
    
    return scaled
    