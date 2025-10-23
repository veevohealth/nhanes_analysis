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
from utils.data_loader import load_and_merge_data
from utils.preprocessing import (
    filter_age, map_sex, calculate_sbp, process_anti_htn_meds,
    process_statin, process_diabetes, process_smoking
)

def get_datasets(suffix: str, config: dict):
    """Returns a dictionary of datasets for the healthy comparator analysis."""
    return {
        "demo": (f"DEMO{suffix}.csv", ["seqn", "riagendr", "ridageyr", config['weight_col']]),
        "bpx": (config['bpx_file'], ["seqn", "bpxosy1", "bpxosy2", "bpxosy3"]),
        "bpq": (f"BPQ{suffix}.csv", ["seqn", "bpq150", "bpq101d"]),
        "tchol": (f"TCHOL{suffix}.csv", ["seqn", "lbxtc"]),
        "hdl": (f"HDL{suffix}.csv", ["seqn", "lbdhdd"]),
        "diq": (f"DIQ{suffix}.csv", ["seqn", "diq010"]),
        "smq": (f"SMQ{suffix}.csv", ["seqn", "smq040"]),
        "mcq": (f"MCQ{suffix}.csv", ["seqn", "mcq300c"]),
        "bmx": (f"BMX{suffix}.csv", ["seqn", "bmxwt", "bmxht"]),
    }

def preprocess_data(df: pd.DataFrame):
    """Preprocesses and cleans the merged NHANES data."""
    df = filter_age(df, min_age=18, max_age=79)
    df = map_sex(df)
    df = calculate_sbp(df)
    df = process_anti_htn_meds(df, fill_na_value=2)
    df = process_statin(df, fill_na_value=2)
    df = process_diabetes(df, fill_na_value=2)
    df = process_smoking(df, fill_na_value=3)

    # Total Cholesterol & HDL
    df.rename(columns={'lbxtc': 'total_cholesterol', 'lbdhdd': 'hdl_c'}, inplace=True)

    # BMI Calculation
    # Formula: weight (kg) / (height (m))^2
    # Height is in cm, so convert to m
    if 'bmxwt' in df.columns and 'bmxht' in df.columns:
        df['bmi'] = df['bmxwt'] / ((df['bmxht'] / 100) ** 2)
    else:
        df['bmi'] = np.nan

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
        'hdl_c': 'hdl_c',
        'bmi': 'bmi',
        'anti_htn_meds': 'anti_htn_meds',
        'statin': 'statin',
        't2dm': 't2dm',
        'smoking': 'smoking',
        'family_history_mi': 'family_history_mi',
        'cycle': 'cycle'
    }
    # Use a list of available columns to avoid KeyError
    available_cols = [col for col in final_cols.keys() if col in df.columns]
    df_final = df[available_cols].rename(columns=final_cols)
    
    # Drop rows with any remaining missing values in key modeling columns
    df_final.dropna(subset=['age', 'sex', 'sbp', 'total_cholesterol', 'hdl_c', 'bmi', 'weight'], inplace=True)
    
    return df_final

def define_healthy_population(df: pd.DataFrame):
    """
    Filters for the 'generally healthy' population.
    - No diabetes
    - Non-smoker
    - Not obese (BMI < 30)
    """
    return df[
        (df['t2dm'] == False) & 
        (df['smoking'] == False) &
        (df['bmi'] < 30)
    ].copy()

def get_comparator_models_and_prevalences(healthy_df: pd.DataFrame):
    """
    Trains regression models for SBP and total cholesterol, and calculates
    prevalences of medication use in the healthy population.
    """
    # Define a low-risk subgroup for training models (no HTN meds, SBP < 140)
    low_risk_df = healthy_df[
        (healthy_df['anti_htn_meds'] == False) &
        (healthy_df['sbp'] < 140)
    ].copy()

    # SBP model
    sbp_train_df = low_risk_df.copy()
    sbp_train_df['is_male'] = (sbp_train_df['sex'] == 'male').astype(int)
    sbp_train_df['age_sex_interaction'] = sbp_train_df['age'] * sbp_train_df['is_male']
    # Spline for women post-menopause, average age 52.
    # Source: https://my.clevelandclinic.org/health/diseases/21841-menopause
    sbp_train_df['female_age_spline'] = (1 - sbp_train_df['is_male']) * np.maximum(0, sbp_train_df['age'] - 52)
    
    X_sbp = sbp_train_df[['age', 'is_male', 'age_sex_interaction', 'female_age_spline']]
    y_sbp = sbp_train_df['sbp']
    weights_sbp = sbp_train_df['weight']
    
    sbp_model = LinearRegression()
    sbp_model.fit(X_sbp, y_sbp, sample_weight=weights_sbp)

    # Total Cholesterol model
    chol_train_df = low_risk_df.copy()
    chol_train_df['is_male'] = (chol_train_df['sex'] == 'male').astype(int)
    chol_train_df['age_sex_interaction'] = chol_train_df['age'] * chol_train_df['is_male']
    chol_train_df['female_age_spline'] = (1 - chol_train_df['is_male']) * np.maximum(0, chol_train_df['age'] - 52)

    X_chol = chol_train_df[['age', 'is_male', 'age_sex_interaction', 'female_age_spline']]
    y_chol = chol_train_df['total_cholesterol']
    weights_chol = chol_train_df['weight']
    
    chol_model = LinearRegression()
    chol_model.fit(X_chol, y_chol, sample_weight=weights_chol)
    
    # HDL Cholesterol model
    hdl_train_df = low_risk_df.copy()
    hdl_train_df['is_male'] = (hdl_train_df['sex'] == 'male').astype(int)
    hdl_train_df['age_sex_interaction'] = hdl_train_df['age'] * hdl_train_df['is_male']
    hdl_train_df['female_age_spline'] = (1 - hdl_train_df['is_male']) * np.maximum(0, hdl_train_df['age'] - 52)

    X_hdl = hdl_train_df[['age', 'is_male', 'age_sex_interaction', 'female_age_spline']]
    y_hdl = hdl_train_df['hdl_c']
    weights_hdl = hdl_train_df['weight']
    
    hdl_model = LinearRegression()
    hdl_model.fit(X_hdl, y_hdl, sample_weight=weights_hdl)

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

    return sbp_model, chol_model, hdl_model, prevalences

def generate_comparator_table(sbp_model: LinearRegression, chol_model: LinearRegression, hdl_model: LinearRegression, prevalences: dict):
    """Generates a table of healthy comparator values for ages 30-79."""
    ages = np.arange(18, 80)
    sexes = ['male', 'female']
    
    results = []
    for sex in sexes:
        for age in ages:
            is_male = 1 if sex == 'male' else 0
            female_age_spline = (1 - is_male) * np.maximum(0, age - 52)
            
            # Predict SBP
            sbp_features = np.array([age, is_male, age * is_male, female_age_spline]).reshape(1, -1)
            pred_sbp = sbp_model.predict(sbp_features)[0]
            
            # Predict Total Cholesterol
            chol_features = np.array([age, is_male, age * is_male, female_age_spline]).reshape(1, -1)
            pred_chol = chol_model.predict(chol_features)[0]
            
            # Predict HDL
            hdl_features = np.array([age, is_male, age * is_male, female_age_spline]).reshape(1, -1)
            pred_hdl = hdl_model.predict(hdl_features)[0]
            
            # Non-HDL Cholesterol
            non_hdl_c = pred_chol - pred_hdl

            results.append({
                'age': age,
                'sex': sex,
                'healthy_sbp': pred_sbp,
                'healthy_total_cholesterol': pred_chol,
                'healthy_hdl_c': pred_hdl,
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

def create_total_cholesterol_plot(df: pd.DataFrame):
    """Creates a line plot of healthy total cholesterol by age and sex."""
    
    sns.set_style("whitegrid")
    g = sns.FacetGrid(df, col="sex", height=6, aspect=1.2, col_order=['female', 'male'])
    g.map(sns.lineplot, "age", "healthy_total_cholesterol", color='mediumseagreen')

    # Customizing the plot
    for i, ax in enumerate(g.axes.flat):
        sex = ['Women', 'Men'][i]
        ax.set_title(f'{sex}')
        ax.set_xlabel("Age, y")
        if i == 0:
            ax.set_ylabel("Predicted Healthy Total Cholesterol, mg/dL")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    g.fig.suptitle("Predicted Healthy Total Cholesterol by Age and Sex (NHANES 2015-2023)", y=1.03)
    
    plt.tight_layout()
    plt.savefig("healthy_total_cholesterol_by_age.png", bbox_inches='tight')
    print("Plot saved to healthy_total_cholesterol_by_age.png")

def create_hdl_cholesterol_plot(df: pd.DataFrame):
    """Creates a line plot of healthy HDL cholesterol by age and sex."""
    
    sns.set_style("whitegrid")
    g = sns.FacetGrid(df, col="sex", height=6, aspect=1.2, col_order=['female', 'male'])
    g.map(sns.lineplot, "age", "healthy_hdl_c", color='goldenrod')

    # Customizing the plot
    for i, ax in enumerate(g.axes.flat):
        sex = ['Women', 'Men'][i]
        ax.set_title(f'{sex}')
        ax.set_xlabel("Age, y")
        if i == 0:
            ax.set_ylabel("Predicted Healthy HDL Cholesterol, mg/dL")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    g.fig.suptitle("Predicted Healthy HDL Cholesterol by Age and Sex (NHANES 2015-2023)", y=1.03)
    
    plt.tight_layout()
    plt.savefig("healthy_hdl_cholesterol_by_age.png", bbox_inches='tight')
    print("Plot saved to healthy_hdl_cholesterol_by_age.png")

def create_non_hdl_cholesterol_plot(df: pd.DataFrame):
    """Creates a line plot of healthy non-HDL cholesterol by age and sex."""
    
    sns.set_style("whitegrid")
    g = sns.FacetGrid(df, col="sex", height=6, aspect=1.2, col_order=['female', 'male'])
    g.map(sns.lineplot, "age", "healthy_non_hdl_c", color='darkorchid')

    # Customizing the plot
    for i, ax in enumerate(g.axes.flat):
        sex = ['Women', 'Men'][i]
        ax.set_title(f'{sex}')
        ax.set_xlabel("Age, y")
        if i == 0:
            ax.set_ylabel("Predicted Healthy Non-HDL Cholesterol, mg/dL")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    g.fig.suptitle("Predicted Healthy Non-HDL Cholesterol by Age and Sex (NHANES 2015-2023)", y=1.03)
    
    plt.tight_layout()
    plt.savefig("healthy_non_hdl_cholesterol_by_age.png", bbox_inches='tight')
    print("Plot saved to healthy_non_hdl_cholesterol_by_age.png")

def main():
    """Main function to run the analysis."""
    script_dir = Path(__file__).parent
    data_root = script_dir / 'data'
    
    year_dirs = ['2021-2023', '2017-2018', '2015-2016']
    
    all_dfs = []
    for year_dir in year_dirs:
        data_path = data_root / year_dir
        print(f"Processing data for {year_dir}...")
        
        year_config = {
            '2021-2023': {'suffix': '_L', 'bpx_file': 'BPXO_L.csv', 'weight_col': 'wtmec2yr'},
            '2017-2018': {'suffix': '_J', 'bpx_file': 'BPXO_J.csv', 'weight_col': 'wtmec2yr'},
            '2015-2016': {'suffix': '_I', 'bpx_file': 'BPX_I.csv', 'weight_col': 'wtmec2yr'},
        }
        config = year_config.get(data_path.name)
        suffix = config['suffix']
        datasets = get_datasets(suffix, config)

        df_merged = load_and_merge_data(data_path, datasets)
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

        sbp_model, chol_model, hdl_model, prevalences = get_comparator_models_and_prevalences(df_healthy)
        print("Trained models and calculated prevalences successfully.")

        comparator_table = generate_comparator_table(sbp_model, chol_model, hdl_model, prevalences)
        
        output_filename = 'healthy_comparator_values.csv'
        comparator_table.to_csv(output_filename, index=False)
        print(f"Saved healthy comparator values to {output_filename}")

        create_sbp_plot(comparator_table)
        create_total_cholesterol_plot(comparator_table)
        create_hdl_cholesterol_plot(comparator_table)
        create_non_hdl_cholesterol_plot(comparator_table)

if __name__ == "__main__":
    main()
