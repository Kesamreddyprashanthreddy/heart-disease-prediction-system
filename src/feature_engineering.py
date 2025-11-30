import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils import (
    setup_logging, 
    ensure_directory, 
    print_dataframe_info,
    save_json
)
from features.encoding import label_encode_features, one_hot_encode_features
from features.scaling import scale_features
from features.polynomial import create_polynomial_features, create_binned_features
from features.derived import create_derived_features

logger = setup_logging()

def engineer_features(df, target_col=None, scaling_method='standard', 
                     test_size=0.2, random_state=42, models_dir="d:/Python/machine-learning-pipeline/models"):
   
    logger.info("Starting feature engineering pipeline...")
    
    ensure_directory(models_dir)
    
    df_processed = df.copy()
    print_dataframe_info(df_processed, "Input Dataset")
    
    if target_col and target_col in df_processed.columns:
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
    else:
        logger.warning("Target column not specified")
        X = df_processed
        y = None
    
    # Initialize transformers
    label_encoders = {}
    onehot_encoder = None
    scaler = None
    all_transformed_features = []
    
    # Apply transformations
    X, label_encoders, features = label_encode_features(X, label_encoders, models_dir=models_dir)
    all_transformed_features.extend(features)
    
    X, onehot_encoder, features = one_hot_encode_features(X, onehot_encoder, models_dir=models_dir)
    all_transformed_features.extend(features)
    
    X, features = create_derived_features(X)
    all_transformed_features.extend(features)
    
    X, features, _ = create_binned_features(X)
    all_transformed_features.extend(features)
    
    X, features = create_polynomial_features(X)
    all_transformed_features.extend(features)
    
    X, scaler, features = scale_features(X, scaler, method=scaling_method, models_dir=models_dir)
    all_transformed_features.extend(features)
    
    print_dataframe_info(X, "Processed Features")
    
    # Save feature info
    feature_info = {
        'original_features': df.columns.tolist(),
        'final_features': X.columns.tolist(),
        'transformed_features': all_transformed_features,
        'feature_count': {
            'original': len(df.columns),
            'final': len(X.columns),
            'added': len(all_transformed_features)
        },
        'transformations_applied': [
            'label_encoding',
            'one_hot_encoding', 
            'derived_features',
            'binned_features',
            'polynomial_features',
            f'{scaling_method}_scaling'
        ]
    }
    
    save_json(feature_info, os.path.join(models_dir, 'feature_info.json'))
    
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if y.dtype == 'object' or y.nunique() <= 10 else None
        )
        
        logger.info(f"Train-test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        logger.info(f"Created {len(X.columns)} features from {len(df.columns)} original columns")
        
        return X_train, X_test, y_train, y_test
    else:
        logger.info(f"Created {len(X.columns)} features from {len(df.columns)} original columns")
        return X, None, None, None

def main():
    # Run feature engineering
    data_path = "d:/Python/machine-learning-pipeline/data/processed/cleaned_data.csv"
    
    if not os.path.exists(data_path):
        logger.warning("Cleaned data not found. Running data cleaning first...")
        from data_cleaning import main as run_cleaning
        run_cleaning()
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    X_train, X_test, y_train, y_test = engineer_features(
        df, 
        target_col='condition',
        scaling_method='standard'
    )
    
    ensure_directory("d:/Python/machine-learning-pipeline/data/processed")
    
    if X_train is not None and X_test is not None:
        X_train.to_csv("d:/Python/machine-learning-pipeline/data/processed/X_train.csv", index=False)
        X_test.to_csv("d:/Python/machine-learning-pipeline/data/processed/X_test.csv", index=False)
        y_train.to_csv("d:/Python/machine-learning-pipeline/data/processed/y_train.csv", index=False)
        y_test.to_csv("d:/Python/machine-learning-pipeline/data/processed/y_test.csv", index=False)
        
        print("\nFeature Engineering Complete!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Created {len(X_train.columns)} features")
    else:
        logger.error("Feature engineering returned None values")

if __name__ == "__main__":
    main()
