# Author: William Liu <liwi@ohsu.edu>

import os
from ._read_txt import read_txt
from ._read_mat import read_mat


def read_raw(file_path: str):
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.txt':
        raw_fnirs = read_txt(file_path)
    elif file_extension == '.mat':
        raw_fnirs = read_mat(file_path)
    else:
        raise TypeError(f"""{filename} provided in {file_extension} format.
        Expected .txt or .mat.""")

    # Check to ensure numerical cols are of dtype float64
    if not all(raw_fnirs['data'].iloc[:, 1:-1].dtypes == 'float64'):
        raise TypeError("Unexpected type found in raw dataframe for "
                        f"{filename}.")

    return raw_fnirs
