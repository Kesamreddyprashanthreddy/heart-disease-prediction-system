import pandas as pd
import numpy as np

def add_ratio_features(df):
    df_new = df.copy()

    df_new['age_to_thalach_ratio'] = np.where(
        df_new['thalach'] != 0,
        df_new['age'] / df_new['thalach'],
        0
    )
    df_new['age_to_ca_ratio'] = np.where(
        df_new['ca'] != 0,
        df_new['age'] / df_new['ca'],
        0
    )
    
    df_new['sex_to_thal_ratio'] = np.where(
        df_new['thal'] != 0,
        df_new['sex'] / df_new['thal'],
        0
    )
    
    return df_new

def add_statistics_features(df):
    df_new = df.copy()
    
    numerical_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                      'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    df_new['row_mean'] = df_new[numerical_cols].mean(axis=1, skipna=True).fillna(0)
    df_new['row_std'] = df_new[numerical_cols].std(axis=1, skipna=True).fillna(0)
    df_new['row_max_min_diff'] = (df_new[numerical_cols].max(axis=1, skipna=True) - 
                                   df_new[numerical_cols].min(axis=1, skipna=True)).fillna(0)
    
    return df_new

def add_binned_features(df):
    df_new = df.copy()

    binning_config = {
        'age': {'bins': [0, 40, 55, 70, 100], 'labels': [0, 1, 2, 3]},
        'trestbps': {'bins': [0, 120, 140, 180, 300], 'labels': [0, 1, 2, 3]},
        'chol': {'bins': [0, 200, 240, 300, 600], 'labels': [0, 1, 2, 3]},
        'thalach': {'bins': [0, 120, 150, 180, 250], 'labels': [0, 1, 2, 3]},
        'oldpeak': {'bins': [-1, 0, 1, 2, 10], 'labels': [0, 1, 2, 3]},
        'age_to_thalach_ratio': {'bins': [0, 0.3, 0.4, 0.5, 10], 'labels': [0, 1, 2, 3]},
        'age_to_ca_ratio': {'bins': [0, 15, 25, 40, 1000], 'labels': [0, 1, 2, 3]},
        'row_mean': {'bins': [0, 5, 7, 9, 20], 'labels': [0, 1, 2, 3]},
        'row_std': {'bins': [0, 3, 5, 8, 20], 'labels': [0, 1, 2, 3]},
        'row_max_min_diff': {'bins': [0, 150, 180, 220, 500], 'labels': [0, 1, 2, 3]}
    }
    
    for column, config in binning_config.items():
        if column in df_new.columns:
            col_data = df_new[column].fillna(0)
            try:
                binned = pd.cut(col_data, bins=config['bins'], 
                              labels=config['labels'], include_lowest=True)
                df_new[f'{column}_binned'] = binned.fillna(0).astype(int)
            except:
                df_new[f'{column}_binned'] = 0
    
    return df_new

def add_squared_features(df):
    df_new = df.copy()
    
    df_new['age_squared'] = df_new['age'] ** 2
    df_new['sex_squared'] = df_new['sex'] ** 2
    df_new['cp_squared'] = df_new['cp'] ** 2
    df_new['trestbps_squared'] = df_new['trestbps'] ** 2
    df_new['chol_squared'] = df_new['chol'] ** 2
    
    return df_new

def add_interaction_features(df):
    df_new = df.copy()
    
    df_new['age_x_sex'] = df_new['age'] * df_new['sex']
    df_new['age_x_cp'] = df_new['age'] * df_new['cp']
    df_new['sex_x_cp'] = df_new['sex'] * df_new['cp']
    df_new['sex_x_trestbps'] = df_new['sex'] * df_new['trestbps']
    df_new['cp_x_trestbps'] = df_new['cp'] * df_new['trestbps']
    df_new['cp_x_chol'] = df_new['cp'] * df_new['chol']
    df_new['trestbps_x_chol'] = df_new['trestbps'] * df_new['chol']
    
    return df_new

def engineer_features(input_data):
    df = pd.DataFrame([input_data])
    df = add_ratio_features(df)
    df = add_statistics_features(df)
    df = add_binned_features(df)
    df = add_squared_features(df)
    df = add_interaction_features(df)
    df = df.fillna(0)
    
    return df
