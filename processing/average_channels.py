# Author: William Liu <liwi@ohsu.edu>

import pandas as pd
import numpy as np


def average_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average channels across each hemisphere and the entire brain (grand).

    :param df: dataframe of processed fnirs, without short channels
    :return: dataframe of averaged channels
    """
    if type(df) != pd.DataFrame:
        raise TypeError(f"Must provide a dataframe, not {type(df)}")

    df_copy = df.copy()
    # Create dictionary of averaged channels
    dict_for_df = {
        'Sample number': df_copy['Sample number'],
        'right oxy': np.mean(df_copy.filter(regex='Rx1-Tx[0-9] O2Hb'), axis=1),
        'right dxy': np.mean(df_copy.filter(regex='Rx1-Tx[0-9] HHb'), axis=1),
        'left oxy': np.mean(df_copy.filter(regex='Rx2-Tx[0-9] O2Hb'), axis=1),
        'left dxy': np.mean(df_copy.filter(regex='Rx2-Tx[0-9] HHb'), axis=1),
        'grand oxy': np.mean(df_copy.filter(regex='O2Hb'), axis=1),
        'grand dxy': np.mean(df_copy.filter(regex='HHb'), axis=1),
        'Event': df_copy['Event']
    }

    ret_df = pd.DataFrame(dict_for_df)

    return ret_df
