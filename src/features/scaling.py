import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from utils import setup_logging, identify_column_types, save_pickle

logger = setup_logging()

def scale_features(df, scaler, method='standard', columns=None, models_dir=None):
    logger.info(f"Applying {method} scaling...")
    
    df_scaled = df.copy()
    col_types = identify_column_types(df_scaled)
    
    if columns is None:
        columns = col_types['numerical']
    
    valid_columns = [col for col in columns if col in df_scaled.columns 
                    and col in col_types['numerical']]
    
    if not valid_columns:
        logger.warning("No numerical columns for scaling")
        return df_scaled, scaler, []
    
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled[valid_columns] = scaler.fit_transform(df_scaled[valid_columns])
        
        if models_dir:
            save_pickle(scaler, os.path.join(models_dir, 'feature_scaler.pkl'))
    else:
        df_scaled[valid_columns] = scaler.transform(df_scaled[valid_columns])
    
    logger.info(f"Scaled {len(valid_columns)} columns using {method} scaling")
    return df_scaled, scaler, valid_columns
