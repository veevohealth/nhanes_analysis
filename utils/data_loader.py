"""
Utility functions for loading NHANES data.
"""
import pandas as pd
from pathlib import Path

def load_and_merge_data(data_path: Path, datasets: dict):
    """Loads and merges the necessary NHANES data files for a specific cycle."""
    
    year_config = {
        '2021-2023': {'suffix': '_L', 'bpx_file': 'BPXO_L.csv', 'weight_col': 'wtmec2yr'},
        '2017-2018': {'suffix': '_J', 'bpx_file': 'BPXO_J.csv', 'weight_col': 'wtmec2yr'},
        '2015-2016': {'suffix': '_I', 'bpx_file': 'BPX_I.csv', 'weight_col': 'wtmec2yr'},
    }
    
    year = data_path.name
    config = year_config.get(year)
    if not config:
        print(f"Warning: No configuration found for {year}. Skipping.")
        return None
        
    df_merged = None
    
    for _, (filename, cols) in datasets.items():
        file_path = data_path / filename
        try:
            df = pd.read_csv(file_path, usecols=cols)
            df.columns = [c.lower() for c in df.columns]
            
            if df_merged is None:
                df_merged = df
            else:
                df_merged = pd.merge(df_merged, df, on="seqn", how="left")
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {data_path}. Skipping.")
        except ValueError as e:
            print(f"Warning: Could not read {filename}. Error: {e}. Skipping.")

    if df_merged is not None:
        df_merged['cycle'] = year
    return df_merged
