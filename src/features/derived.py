import pandas as pd
import numpy as np
from utils import setup_logging, identify_column_types

logger = setup_logging()

def create_derived_features(df):
    logger.info("Creating derived features...")
    
    df_derived = df.copy()
    col_types = identify_column_types(df_derived)
    numerical_cols = col_types['numerical']
    categorical_cols = col_types['categorical']
    
    transformed_features = []
    ratio_pairs = []
    for i, col1 in enumerate(numerical_cols):
        for col2 in numerical_cols[i+1:]:
            if col1 != col2:
                try:
                    correlation = df_derived[col1].corr(df_derived[col2])
                    if abs(correlation) > 0.3:
                        ratio_pairs.append((col1, col2))
                except:
                    pass
    
    for col1, col2 in ratio_pairs[:3]:
        try:
            denominator = df_derived[col2].replace(0, np.nan)
            feature_name = f'{col1}_to_{col2}_ratio'
            df_derived[feature_name] = df_derived[col1] / denominator
            transformed_features.append(feature_name)
        except:
            pass
    
    if len(numerical_cols) >= 3:
        try:
            feature_name = 'row_mean'
            df_derived[feature_name] = df_derived[numerical_cols].mean(axis=1)
            transformed_features.append(feature_name)
            
            feature_name = 'row_std'
            df_derived[feature_name] = df_derived[numerical_cols].std(axis=1)
            transformed_features.append(feature_name)
            
            feature_name = 'row_max_min_diff'
            df_derived[feature_name] = (df_derived[numerical_cols].max(axis=1) - 
                                      df_derived[numerical_cols].min(axis=1))
            transformed_features.append(feature_name)
        except:
            pass
    
    if len(categorical_cols) >= 2:
        try:
            col1, col2 = categorical_cols[0], categorical_cols[1]
            feature_name = f'{col1}_{col2}_combined'
            df_derived[feature_name] = (df_derived[col1].astype(str) + '_' + 
                                      df_derived[col2].astype(str))
            transformed_features.append(feature_name)
        except:
            pass
    
    for col in categorical_cols[:2]:
        try:
            feature_name = f'{col}_frequency'
            frequency_map = df_derived[col].value_counts().to_dict()
            df_derived[feature_name] = df_derived[col].map(frequency_map)
            transformed_features.append(feature_name)
        except:
            pass
    
    logger.info(f"Created {len(transformed_features)} derived features")
    return df_derived, transformed_features
