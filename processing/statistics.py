# Author: William Liu <liwi@ohsu.edu>

import math
import os
import pandas as pd
import numpy as np


def calculate_statistics(segments: dict, file: str) -> pd.DataFrame:
    """
    Calculate statistics for each segment. Metrics include mean, median,
    standard deviation, range, and mean of the detrended time series.
    Based on variability measures described in Maidan et. al. 2022, Neural
    Variability in the Prefrontal Cortex as a Reflection of Neural Flexibility
    and Stability in Patients With Parkison Disease.

    :param segments: dictionary of processed fNIRS data, split into segments
    :param file: path to data file
    :return: dataframe of statistics, calculted for each segment
    """
    # Get filename
    name = os.path.basename(file)

    # Initialize dict to store calculations
    data_as_dict = dict()

    for seg, df in segments.items():
        if seg == 'Quiet Stance':
            continue
        for col in list(df.columns):
            if 'Sample number' in col or 'Event' in col:
                continue
            label = seg + ' ' + col
            values = np.array(df[col], dtype=np.float64)
            mean = math.fsum(values) / len(values)
            median = np.median(values)
            std = np.std(values, dtype=np.float64)
            r = np.ptp(values)
            detrended = detrended_mean(values)
            # Populate dictionary
            data_as_dict[label + ' Mean'] = mean
            data_as_dict[label + ' Median'] = median
            data_as_dict[label + ' StDev'] = std
            data_as_dict[label + ' Range'] = r
            data_as_dict[label + ' Detrended'] = detrended

    ret_df = pd.DataFrame(data=data_as_dict, index=[name])

    return ret_df


def detrended_mean(data):
    """
    Calculate the mean of the detrended time series. Similar to the derivative
    of a continuous signal and reduced local trends in the signals. The higher
    the mean the more variable the signal. See Maidan et. al. 2022.
    1. First create a new time series consisting of every xth sample, where x
    is the sample rate.
    2. Find the first difference of this new time series.
    3. Take the mean
    """
    # Step 1. Get every 50th sample
    x = data[::50].copy()
    # Step 2. First difference
    diff = np.diff(x)
    # Step 3. Take the mean
    mean = math.fsum(diff) / len(diff)

    return mean
