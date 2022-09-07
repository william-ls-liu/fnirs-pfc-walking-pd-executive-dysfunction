# Author: William Liu <liwi@ohsu.edu>

import pandas as pd
import numpy as np
import math

def calculate_statistics(segments: dict) -> pd.DataFrame:
    """
    Calculate statistics for each segment. Metrics include mean, median,
    standard deviation, range, and mean of the detrended time series.
    Based on variability measures described in Maidan et. al. 2022, Neural
    Variability in the Prefrontal Cortex as a Reflection of Neural Flexibility
    and Stability in Patients With Parkison Disease.

    :param segments: dictionary of processed fNIRS data, split into segments
    :return: dataframe of statistics, calculted for each segment
    """
    # Store iterables used to creates a pandas hierarchical index later on
    multi_index = dict()
    multi_index['Metrics'] = ['Mean', 'Median', 'StDev', 'Range', 'Detrended']
    # Dict to store the metrics, used to build final dataframe
    metrics = dict()
    for key, val in segments.items():
        if 'Segment' in multi_index:
            multi_index['Segment'].append(key)
        else:
            multi_index['Segment'] = [key]

        for col in list(val.columns):
            if 'Sample number' in col or 'Event' in col:
                continue
            data = np.array(val[col], dtype=np.float64)
            mean = math.fsum(data) / len(data)
            median = np.median(data)
            std = np.std(data, dtype=np.float64)
            r = np.ptp(data)
            detrended = detrended_mean(data)
            metrics_as_list = [mean, median, std, r, detrended]
            if col in metrics:
                metrics[col].extend(metrics_as_list)
            else:
                metrics[col] = metrics_as_list

    index = pd.MultiIndex.from_product([multi_index['Segment'], multi_index['Metrics']], names=['Segment', 'Metric'])
    df = pd.DataFrame(data=metrics, index=index)

    return df

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
