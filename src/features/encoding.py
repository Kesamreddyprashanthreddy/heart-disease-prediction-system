import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
from utils import setup_logging, identify_column_types, save_pickle

logger = setup_logging()

def label_encode_features(df, label_encoders, columns=None, models_dir=None):
    logger.info("Applying label encoding...")
    
    df_encoded = df.copy()
    col_types = identify_column_types(df_encoded)
    
    if columns is None:
        columns = [col for col in col_types['categorical'] 
                  if df_encoded[col].nunique() > 10]
    
    if not columns:
        logger.warning("No suitable columns for label encoding")
        return df_encoded, label_encoders, []
    
    valid_columns = [col for col in columns if col in df_encoded.columns 
                    and col in col_types['categorical']]
    
    transformed_features = []
    for col in valid_columns:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            df_encoded[f'{col}_encoded'] = label_encoders[col].fit_transform(
                df_encoded[col].astype(str)
            )
            
            if models_dir:
                save_pickle(label_encoders[col],
                           os.path.join(models_dir, f'label_encoder_{col}.pkl'))
        else:
            df_encoded[f'{col}_encoded'] = label_encoders[col].transform(
                df_encoded[col].astype(str)
            )
        
        transformed_features.append(f'{col}_encoded')
    
    logger.info(f"Label encoded {len(valid_columns)} columns")
    return df_encoded, label_encoders, transformed_features

def one_hot_encode_features(df, onehot_encoder, columns=None, drop_first=True, 
                            max_categories=10, models_dir=None):
    logger.info("Applying one-hot encoding...")
    
    df_encoded = df.copy()
    col_types = identify_column_types(df_encoded)
    
    if columns is None:
        columns = [col for col in col_types['categorical'] 
                  if df_encoded[col].nunique() <= max_categories]
    
    if not columns:
        logger.warning("No suitable columns for one-hot encoding")
        return df_encoded, onehot_encoder, []
    
    valid_columns = [col for col in columns if col in df_encoded.columns 
                    and col in col_types['categorical']]
    
    if not valid_columns:
        logger.warning("No valid categorical columns found")
        return df_encoded, onehot_encoder, []
    
    if onehot_encoder is None:
        onehot_encoder = OneHotEncoder(
            drop='first' if drop_first else None,
            sparse_output=False,
            handle_unknown='ignore'
        )
        
        encoded_array = onehot_encoder.fit_transform(df_encoded[valid_columns])
        
        if models_dir:
            save_pickle(onehot_encoder, 
                       os.path.join(models_dir, 'onehot_encoder.pkl'))
    else:
        encoded_array = onehot_encoder.transform(df_encoded[valid_columns])
    
    feature_names = onehot_encoder.get_feature_names_out(valid_columns)
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_encoded.index)
    
    df_encoded = df_encoded.drop(columns=valid_columns)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    logger.info(f"One-hot encoded {len(valid_columns)} columns into {len(feature_names)} features")
    return df_encoded, onehot_encoder, feature_names.tolist()
