# Author: William Liu <liwi@ohsu.edu>

import pandas as pd
import os
from . import _read_txt
from ._read_txt import read_txt

def read_raw(file_path: str):
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.txt':
        raw_fnirs = read_txt(file_path)
    elif file_extension == '.mat':
        pass
    else:
        raise TypeError(f"File provided in {file_extension} format. Expected .txt or .mat.")
    
    return raw_fnirs
