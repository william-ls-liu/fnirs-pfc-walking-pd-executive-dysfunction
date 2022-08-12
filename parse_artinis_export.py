import pandas as pd

def parse_artinis_export(file_path: str) -> pd.DataFrame:
    """Read the exported file from Artinis fNIRS software and parse into a pandas DataFrame."""
    with open(file_path, 'r') as f:
        lines = f.read().split('\n')
    
    return lines

print(parse_artinis_export('fnirs-pfc-walking-pd-executive-dysfunction/test_data/Turn_129_LongWalk_ST.txt'))
