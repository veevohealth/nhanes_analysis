"""
NHANES Data Analysis for "Generally Healthy Comparator" Calculation.

This script processes NHANES 2015-2023 data to create age- and sex-specific
"generally healthy comparators" based on the methodology described by Blaha et al.
in https://www.ahajournals.org/doi/10.1161/JAHA.120.019351.

The healthy comparator is defined as being free of diabetes and not currently smoking.
Models are trained on this subpopulation to predict SBP and total cholesterol.
Prevalences of medication use are also calculated from this group.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

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
        "mcq": (f"MCQ{suffix}.csv", ["seqn", "mcq300c"]),
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
    
    # Anti-hypertensive meds: 1 -> True, otherwise False.
    df['anti_htn_meds'] = df['bpq150'].fillna(2) == 1

    # Statin: 1 -> True, otherwise False.
    df['statin'] = df['bpq101d'].fillna(2) == 1

    # Total Cholesterol
    df.rename(columns={'lbxtc': 'total_cholesterol'}, inplace=True)

    # Diabetes: 1 (Yes) or 3 (Borderline) -> True.
    df['t2dm'] = df['diq010'].fillna(2).isin([1, 3])

    # Smoking: 1 (every day) or 2 (some days) -> True.
    df['smoking'] = df['smq040'].fillna(3).isin([1, 2])
    
    # Family history of MI: 1 -> True
    if 'mcq300c' in df.columns:
        df['family_history_mi'] = df['mcq300c'].fillna(2) == 1
    else:
        df['family_history_mi'] = False

    # Select and rename final columns
    final_cols = {
        'seqn': 'seqn',
        'ridageyr': 'age',
        'wtmec2yr': 'weight',
        'sex': 'sex',
        'sbp': 'sbp',
        'total_cholesterol': 'total_cholesterol',
        'anti_htn_meds': 'anti_htn_meds',
        'statin': 'statin',
        't2dm': 't2dm',
        'smoking': 'smoking',
        'family_history_mi': 'family_history_mi',
        'cycle': 'cycle'
    }
    df_final = df[list(final_cols.keys())].rename(columns=final_cols)
    
    # Drop rows with any remaining missing values in key modeling columns
    df_final.dropna(subset=['age', 'sex', 'sbp', 'total_cholesterol', 'weight'], inplace=True)
    
    return df_final

def define_healthy_population(df: pd.DataFrame):
    """Filters for the 'generally healthy' population (no diabetes, non-smoker)."""
    return df[(df['t2dm'] == False) & (df['smoking'] == False)].copy()

def get_comparator_models_and_prevalences(healthy_df: pd.DataFrame):
    """
    Trains regression models for SBP and total cholesterol, and calculates
    prevalences of medication use in the healthy population.
    """
    # SBP model (trained on healthy individuals not on anti-HTN meds)
    sbp_train_df = healthy_df[healthy_df['anti_htn_meds'] == False].copy()
    sbp_train_df['is_male'] = (sbp_train_df['sex'] == 'male').astype(int)
    sbp_train_df['age_sex_interaction'] = sbp_train_df['age'] * sbp_train_df['is_male']
    
    X_sbp = sbp_train_df[['age', 'is_male', 'age_sex_interaction']]
    y_sbp = sbp_train_df['sbp']
    weights_sbp = sbp_train_df['weight']
    
    sbp_model = LinearRegression()
    sbp_model.fit(X_sbp, y_sbp, sample_weight=weights_sbp)

    # Total Cholesterol model
    chol_train_df = healthy_df.copy()
    chol_train_df['is_male'] = (chol_train_df['sex'] == 'male').astype(int)
    chol_train_df['age_sex_interaction'] = chol_train_df['age'] * chol_train_df['is_male']
    chol_train_df['female_age_spline'] = (1 - chol_train_df['is_male']) * np.maximum(0, chol_train_df['age'] - 55)

    X_chol = chol_train_df[['age', 'is_male', 'age_sex_interaction', 'female_age_spline']]
    y_chol = chol_train_df['total_cholesterol']
    weights_chol = chol_train_df['weight']
    
    chol_model = LinearRegression()
    chol_model.fit(X_chol, y_chol, sample_weight=weights_chol)
    
    # Prevalences using weighted average
    def weighted_prevalence(df, column, weights_col):
        if df[weights_col].sum() == 0:
            return 0
        return np.average(df[column], weights=df[weights_col])

    prev_anti_htn = weighted_prevalence(healthy_df, 'anti_htn_meds', 'weight')
    prev_statin = weighted_prevalence(healthy_df, 'statin', 'weight')
    prev_fam_hist = weighted_prevalence(healthy_df, 'family_history_mi', 'weight')

    prevalences = {
        'anti_htn_meds': prev_anti_htn,
        'statin': prev_statin,
        'family_history_mi': prev_fam_hist
    }

    return sbp_model, chol_model, prevalences

def generate_comparator_table(sbp_model: LinearRegression, chol_model: LinearRegression, prevalences: dict):
    """Generates a table of healthy comparator values for ages 30-79."""
    ages = np.arange(30, 80)
    sexes = ['male', 'female']
    
    results = []
    for sex in sexes:
        for age in ages:
            is_male = 1 if sex == 'male' else 0
            
            # Predict SBP
            sbp_features = np.array([age, is_male, age * is_male]).reshape(1, -1)
            pred_sbp = sbp_model.predict(sbp_features)[0]
            
            # Predict Total Cholesterol
            female_age_spline = (1 - is_male) * np.maximum(0, age - 55)
            chol_features = np.array([age, is_male, age * is_male, female_age_spline]).reshape(1, -1)
            pred_chol = chol_model.predict(chol_features)[0]
            
            # HDL
            hdl = 45 if sex == 'male' else 55
            
            # Non-HDL Cholesterol
            non_hdl_c = pred_chol - hdl

            results.append({
                'age': age,
                'sex': sex,
                'healthy_sbp': pred_sbp,
                'healthy_total_cholesterol': pred_chol,
                'healthy_hdl_c': hdl,
                'healthy_non_hdl_c': non_hdl_c,
                'prevalence_anti_htn_meds': prevalences['anti_htn_meds'],
                'prevalence_statin': prevalences['statin'],
                'prevalence_family_history_mi': prevalences['family_history_mi']
            })
            
    return pd.DataFrame(results)

def create_sbp_plot(df: pd.DataFrame):
    """Creates a line plot of healthy SBP by age and sex."""
    
    sns.set_style("whitegrid")
    g = sns.FacetGrid(df, col="sex", height=6, aspect=1.2, col_order=['female', 'male'])
    g.map(sns.lineplot, "age", "healthy_sbp", color='cornflowerblue')

    # Customizing the plot
    for i, ax in enumerate(g.axes.flat):
        sex = ['Women', 'Men'][i]
        ax.set_title(f'{sex}')
        ax.set_xlabel("Age, y")
        if i == 0:
            ax.set_ylabel("Predicted Healthy SBP, mmHg")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    g.fig.suptitle("Predicted Healthy SBP by Age and Sex (NHANES 2015-2023)", y=1.03)
    
    plt.tight_layout()
    plt.savefig("healthy_sbp_by_age.png", bbox_inches='tight')
    print("Plot saved to healthy_sbp_by_age.png")

def main():
    """Main function to run the analysis."""
    script_dir = Path(__file__).parent
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
        
        df_healthy = define_healthy_population(df_processed)
        print(f"Defined 'healthy' population: {len(df_healthy)} individuals.")

        sbp_model, chol_model, prevalences = get_comparator_models_and_prevalences(df_healthy)
        print("Trained models and calculated prevalences successfully.")

        comparator_table = generate_comparator_table(sbp_model, chol_model, prevalences)
        
        output_filename = 'healthy_comparator_values.csv'
        comparator_table.to_csv(output_filename, index=False)
        print(f"Saved healthy comparator values to {output_filename}")

        create_sbp_plot(comparator_table)

if __name__ == "__main__":
    main()
