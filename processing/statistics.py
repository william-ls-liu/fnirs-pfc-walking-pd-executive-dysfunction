# Author: William Liu <liwi@ohsu.edu>

import pandas as pd
import numpy as np
import math

def calculate_statistics(segments: dict) -> pd.DataFrame:
    """
    Calculate statistics for each segment. Metrics include mean, median,
    standard deviation, range, and mean of the detrended time series.
    Based on variability measures described in Maidan et. al. 2022, Neural
    Variability in the Prefrontal Cortex as a Reflection fo Neural Flexibility
    and Stability in Patients With Parkison Disease.

    :param segments: dictionary of processed fNIRS data, split into segments
    :return: dataframe of statistics, calculted for each segment
    """
    

