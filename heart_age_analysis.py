"""
NHANES Data Analysis for Heart Age Calculation.

This script processes NHANES 2021-2023 data to calculate heart age based on the PREVENT equations.

Assumptions for missing data:
- Smoking: Assumed to be False if data is missing.
- Anti-hypertensive medication: Assumed to be False if data is missing.
- Statin medication: Assumed to be False if data is missing.
- Diabetes: Assumed to be False if data is missing.
- Serum Creatinine (for eGFR): If missing, eGFR is assumed to be 90 (healthy).
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# IMPORTANT: Add the path to the beta-app-backend directory so we can import the service
sys.path.append('/Users/arvind/veevohealth/beta-app-backend/beta-app-backend')


# Add a check for plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Plotting libraries not found. Please install matplotlib and seaborn.")
    plt = None
    sns = None

# Assuming heart_age_service is in the path.
# This might need adjustment based on where you run the script from.
from app.services.heart_age_service import Profile, prevent_heart_age, prevent_ascvd_risk_10y_base

def load_and_merge_data(data_path: Path):
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
        
    suffix = config['suffix']
    
    datasets = {
        "demo": (f"DEMO{suffix}.csv", ["seqn", "riagendr", "ridageyr", config['weight_col']]),
        "bpx": (config['bpx_file'], ["seqn", "bpxosy1", "bpxosy2", "bpxosy3"]),
        "bpq": (f"BPQ{suffix}.csv", ["seqn", "bpq150", "bpq101d"]),
        "tchol": (f"TCHOL{suffix}.csv", ["seqn", "lbxtc"]),
        "hdl": (f"HDL{suffix}.csv", ["seqn", "lbdhdd"]),
        "diq": (f"DIQ{suffix}.csv", ["seqn", "diq010"]),
        "smq": (f"SMQ{suffix}.csv", ["seqn", "smq040"]),
        "biopro": (f"BIOPRO{suffix}.csv", ["seqn", "lbxscr"]),
    }
    
    df_merged = None
    
    for name, (filename, cols) in datasets.items():
        file_path = data_path / filename
        try:
            df = pd.read_csv(file_path, usecols=cols)
            df.columns = [c.lower() for c in df.columns]
            
            if df_merged is None:
                df_merged = df
            else:
                df_merged = pd.merge(df_merged, df, on="seqn", how="left")
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping.")
        except ValueError as e:
            print(f"Warning: Could not read {filename}. Error: {e}. Skipping.")

    df_merged['cycle'] = year
    return df_merged

def preprocess_data(df: pd.DataFrame):
    """Preprocesses and cleans the merged NHANES data."""
    
    # Age filter (30-79 years)
    df = df[df['ridageyr'].between(30, 79)].copy()
    
    # Sex: 1 -> 'male', 2 -> 'female'
    df['sex'] = df['riagendr'].map({1: 'male', 2: 'female'})
    
    # SBP: average of the 3 readings
    sbp_cols = ['bpxosy1', 'bpxosy2', 'bpxosy3']
    df['sbp'] = df[sbp_cols].mean(axis=1)
    
    # Anti-hypertensive meds: 1 -> True, otherwise False. Missing values are treated as False.
    df['anti_htn_meds'] = df['bpq150'].fillna(0) == 1

    # Non-HDL cholesterol
    df['non_hdl_c'] = df['lbxtc'] - df['lbdhdd']
    df.rename(columns={'lbdhdd': 'hdl_c'}, inplace=True)

    # Diabetes: 1 (Yes) or 3 (Borderline) -> True. Missing values are treated as False.
    df['t2dm'] = df['diq010'].fillna(0).isin([1, 3])

    # Smoking: 1 or 2 -> True, otherwise False. Missing values are treated as False.
    df['smoking'] = df['smq040'].fillna(0).isin([1, 2])

    # Statin: 1 -> True, otherwise False. Missing values are treated as False.
    df['statin'] = df['bpq101d'].fillna(0) == 1

    # eGFR calculation (CKD-EPI 2021)
    # https://www.hepatitisb.uw.edu/page/clinical-calculators/mdrd
    def calculate_egfr(row):
        scr = row['lbxscr']
        age = row['ridageyr']
        is_female = row['sex'] == 'female'
        
        if pd.isna(scr) or pd.isna(age):
            return 90
            
        kappa = 0.7 if is_female else 0.9
        alpha = -0.241 if is_female else -0.302
        
        egfr = 142 * (min(scr / kappa, 1)**alpha) * (max(scr / kappa, 1)**-1.200) * (0.9938**age)
        if is_female:
            egfr *= 1.012
        return egfr

    df['egfr'] = df.apply(calculate_egfr, axis=1)

    # Select and rename final columns
    final_cols = {
        'seqn': 'seqn',
        'ridageyr': 'age',
        'wtmec2yr': 'weight',
        'sex': 'sex',
        'sbp': 'sbp',
        'anti_htn_meds': 'anti_htn_meds',
        'non_hdl_c': 'non_hdl_c',
        'hdl_c': 'hdl_c',
        'statin': 'statin',
        't2dm': 't2dm',
        'smoking': 'smoking',
        'egfr': 'egfr',
        'cycle': 'cycle'
    }
    df_final = df[list(final_cols.keys())].rename(columns=final_cols)
    
    # Drop rows with any remaining missing values in the final columns
    df_final.dropna(inplace=True)
    
    return df_final

def calculate_heart_ages(df: pd.DataFrame):
    """Calculates heart age for each individual in the dataframe."""
    
    results = []
    for _, row in df.iterrows():
        profile = Profile(
            sex=row['sex'],
            sbp=row['sbp'],
            anti_htn_meds=row['anti_htn_meds'],
            non_hdl_c=row['non_hdl_c'],
            hdl_c=row['hdl_c'],
            statin=row['statin'],
            t2dm=row['t2dm'],
            smoking=row['smoking'],
            egfr=row['egfr'],
        )
        
        heart_age_result = prevent_heart_age(
            person_age=row['age'],
            person_profile=profile,
            risk_fn=prevent_ascvd_risk_10y_base,
            risk_kwargs={},
        )
        results.append(heart_age_result)
        
    heart_age_df = pd.DataFrame(results)
    
    return pd.concat([df.reset_index(drop=True), heart_age_df], axis=1)


def group_and_aggregate(df: pd.DataFrame):
    """Groups data by age, sex, and discordance, and aggregates population counts."""
    
    # Create age groups
    age_bins = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    age_labels = [f"{i}-{i+4}" for i in range(30, 80, 5)]
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # Create discordance categories
    conditions = [
        (df['delta_years'] < -5),
        (df['delta_years'].between(-5, 5)),
        (df['delta_years'].between(5, 10, inclusive='right')),
        (df['delta_years'] > 10)
    ]
    categories = ['>5 y younger', 'Within 5 y', '5-10 y older', '>10 y older']
    df['discordance_category'] = np.select(conditions, categories, default='Other')

    # Group and aggregate
    df_agg = df.groupby(['sex', 'age_group', 'discordance_category'])['weight'].sum().reset_index()
    
    # Convert to thousands
    df_agg['population_thousands'] = df_agg['weight'] / 1_000

    return df_agg

def create_plot(df_agg: pd.DataFrame):
    """Creates a stacked bar chart of heart age discordance."""
    
    if plt is None or sns is None:
        return
        
    category_order = ['>10 y older', '5-10 y older', 'Within 5 y', '>5 y younger']
    colors = ['#8B0000', '#FFA500', '#90EE90', '#006400'] # dark red, orange, light green, dark green

    g = sns.FacetGrid(df_agg, col="sex", height=6, aspect=1.2, col_order=['female', 'male'])

    def plot_stacked_bar(data, **kwargs):
        ax = plt.gca()
        pivot_df = data.pivot(index='age_group', columns='discordance_category', values='population_thousands')
        pivot_df = pivot_df.reindex(columns=category_order)
        pivot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)

    g.map_dataframe(plot_stacked_bar)

    # Customizing the plot
    for i, ax in enumerate(g.axes.flat):
        sex = ['Women', 'Men'][i]
        ax.set_title(f'{chr(65+i)} {sex}')
        ax.set_xlabel("Age group, y")
        if i == 0:
            ax.set_ylabel("Individuals, No. (thousands)")
        
        # Reformat y-axis ticks to be more readable with the "(thousands)" label
        yticks = ax.get_yticks()
        ax.set_yticklabels([int(y/1000) for y in yticks])
        
        ax.tick_params(axis='x', rotation=45)
        ax.legend().set_visible(False) # Remove individual legends

    # Add a single legend
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    fig = g.fig
    fig.legend(handles, labels, title='Mean age discordance', bbox_to_anchor=(0.9, 0.85))
    
    fig.suptitle("Discordance Between Chronological Age on NHANES Data", y=1.05)
    
    plt.tight_layout()
    plt.savefig("heart_age_discordance.png", bbox_inches='tight')
    print("Plot saved to heart_age_discordance.png")


def main():
    # Get the directory of the current script.
    script_dir = Path(__file__).parent
    # Path to the data directory, assuming it's a sibling of the script's directory.
    data_root = script_dir / 'data'
    
    year_dirs = ['2021-2023', '2017-2018', '2015-2016']
    
    all_dfs = []
    for year_dir in year_dirs:
        data_path = data_root / year_dir
        print(f"Processing data for {year_dir}...")
        df_merged = load_and_merge_data(data_path)
        if df_merged is not None:
            all_dfs.append(df_merged)

    if not all_dfs:
        print("No data was loaded. Exiting.")
        return

    df_combined = pd.concat(all_dfs, ignore_index=True)

    if df_combined is not None:
        print("All data loaded and merged successfully.")
        df_processed = preprocess_data(df_combined)
        print("Data preprocessed successfully.")
        
        df_with_heart_age = calculate_heart_ages(df_processed)
        print("Heart age calculated successfully.")
        
        df_agg = group_and_aggregate(df_with_heart_age)
        print("Data grouped and aggregated successfully.")
        
        create_plot(df_agg)


if __name__ == "__main__":
    main()
