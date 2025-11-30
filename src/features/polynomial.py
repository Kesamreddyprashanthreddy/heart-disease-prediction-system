import pandas as pd
import numpy as np
from utils import setup_logging, identify_column_types

logger = setup_logging()

def create_polynomial_features(df, columns=None, max_features=5):
    logger.info("Creating polynomial features...")
    
    df_poly = df.copy()
    col_types = identify_column_types(df_poly)
    
    if columns is None:
        columns = col_types['numerical'][:max_features]
    
    valid_columns = [col for col in columns if col in df_poly.columns 
                    and col in col_types['numerical']][:max_features]
    
    if len(valid_columns) < 2:
        logger.warning("Need at least 2 numerical columns")
        return df_poly, []
    
    transformed_features = []
    
    for col in valid_columns[:5]:
        feature_name = f'{col}_squared'
        df_poly[feature_name] = df_poly[col] ** 2
        transformed_features.append(feature_name)
    
    if len(valid_columns) >= 2:
        for i in range(len(valid_columns)):
            for j in range(i+1, min(i+3, len(valid_columns))):
                col1, col2 = valid_columns[i], valid_columns[j]
                feature_name = f'{col1}_x_{col2}'
                df_poly[feature_name] = df_poly[col1] * df_poly[col2]
                transformed_features.append(feature_name)
    
    logger.info(f"Created {len(transformed_features)} polynomial features")
    return df_poly, transformed_features

def create_binned_features(df, columns=None, n_bins=5, strategy='quantile'):
    logger.info("Creating binned features...")
    
    df_binned = df.copy()
    col_types = identify_column_types(df_binned)
    
    if columns is None:
        columns = [col for col in col_types['numerical'] 
                  if df_binned[col].nunique() > n_bins]
    
    valid_columns = [col for col in columns if col in df_binned.columns 
                    and col in col_types['numerical']]
    
    transformed_features = []
    feature_info = {}
    
    for col in valid_columns:
        try:
            if strategy == 'uniform':
                bins = np.linspace(df_binned[col].min(), df_binned[col].max(), n_bins + 1)
            elif strategy == 'quantile':
                bins = df_binned[col].quantile(np.linspace(0, 1, n_bins + 1)).values
                bins = np.unique(bins)
            else:
                raise ValueError(f"Unknown binning strategy: {strategy}")
            
            if len(bins) > 1:
                feature_name = f'{col}_binned'
                df_binned[feature_name] = pd.cut(df_binned[col], bins=bins, 
                                               labels=False, include_lowest=True)
                transformed_features.append(feature_name)
                
                feature_info[feature_name] = {
                    'original_column': col,
                    'bins': bins.tolist(),
                    'strategy': strategy
                }
        
        except Exception as e:
            logger.warning(f"Could not bin column {col}: {str(e)}")
    
    logger.info(f"Created {len(transformed_features)} binned features")
    return df_binned, transformed_features, feature_info
