"""
Utility functions for preprocessing NHANES data.
"""
import pandas as pd
import numpy as np

def filter_age(df: pd.DataFrame, min_age: int = 30, max_age: int = 79) -> pd.DataFrame:
    """Filters the dataframe for a specific age range."""
    return df[df['ridageyr'].between(min_age, max_age)].copy()

def map_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Maps sex from numeric codes to 'male' or 'female' strings."""
    df['sex'] = df['riagendr'].map({1: 'male', 2: 'female'})
    return df

def calculate_sbp(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the average SBP from three readings."""
    sbp_cols = ['bpxosy1', 'bpxosy2', 'bpxosy3']
    df['sbp'] = df[sbp_cols].mean(axis=1)
    return df

def process_anti_htn_meds(df: pd.DataFrame, fill_na_value=0) -> pd.DataFrame:
    """Processes anti-hypertensive medication status."""
    df['anti_htn_meds'] = df['bpq150'].fillna(fill_na_value) == 1
    return df

def process_statin(df: pd.DataFrame, fill_na_value=0) -> pd.DataFrame:
    """Processes statin medication status."""
    df['statin'] = df['bpq101d'].fillna(fill_na_value) == 1
    return df

def process_diabetes(df: pd.DataFrame, fill_na_value=0) -> pd.DataFrame:
    """Processes diabetes status."""
    df['t2dm'] = df['diq010'].fillna(fill_na_value).isin([1, 3])
    return df

def process_smoking(df: pd.DataFrame, fill_na_value=0) -> pd.DataFrame:
    """Processes smoking status."""
    df['smoking'] = df['smq040'].fillna(fill_na_value).isin([1, 2])
    return df
