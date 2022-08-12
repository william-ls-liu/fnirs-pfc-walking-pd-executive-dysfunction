import pandas as pd

def parse_artinis_export(file_path: str) -> pd.DataFrame:
    """Read the exported file from Artinis fNIRS software and parse into a pandas DataFrame."""
    with open(file_path, 'r') as f:
        # Split the .txt file into lines
        lines = f.read().split('\n')
        # Split each line into 'columns'
        rows = [[i for i in j.split('\t')] for j in lines]

        # Get column labels to use for DataFrame
        start = None
        end = None
        for idx, row in enumerate(rows):
            if "(Sample number)" in row:
                start = idx
            elif "(Event)" in row:
                end = idx
                break
        col_labels = rows[start: (end + 1)]
        col_labels = [i[1] for i in col_labels]

        # Remove extra characters from column labels
        for idx, label in enumerate(col_labels):
            if "O2Hb" in label or "HHb" in label:
                new_label = label.split('(')[0].rstrip()
                col_labels[idx] = new_label
            elif "(Sample number)" in label or "(Event)" in label:
                new_label = label.split('(')[1].split(')')[0]
                col_labels[idx] = new_label
            else:
                raise KeyError(f"Unexpected value found in column labels: {label}")

    return col_labels