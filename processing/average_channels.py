# Author: William Liu <liwi@ohsu.edu>

import pandas as pd


def average_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Average channels across each hemisphere and the entire brain (grand)."""
    if type(df) != pd.DataFrame:
        raise TypeError(f"Must provide a dataframe, not {type(df)}")