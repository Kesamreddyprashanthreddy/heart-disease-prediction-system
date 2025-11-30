import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import train_test_split
from utils import (
    setup_logging, 
    ensure_directory, 
    identify_column_types,
    save_pickle,
    load_pickle,
    print_dataframe_info
)

logger = setup_logging()

class FeatureEngineer:
    
    def __init__(self, models_dir="d:/Python/machine-learning-pipeline/models"):
        self.models_dir = models_dir
        ensure_directory(models_dir)
        
        self.transformers = {
            'scaler': None,
            'label_encoders': {},
            'onehot_encoder': None,
            'ordinal_encoder': None
        }
        
        self.feature_info = {}
        self.transformed_features = []
    
    def one_hot_encode_features(self, df, columns=None, drop_first=True, max_categories=10):
        logger.info("Applying one-hot encoding...")
        
        df_encoded = df.copy()
        col_types = identify_column_types(df_encoded)
        
        if columns is None:
            columns = [col for col in col_types['categorical'] 
                      if df_encoded[col].nunique() <= max_categories]
        
        if not columns:
            logger.warning("No suitable columns found for one-hot encoding")
            return df_encoded
        
        valid_columns = [col for col in columns if col in df_encoded.columns 
                        and col in col_types['categorical']]
        
        if not valid_columns:
            logger.warning("No valid categorical columns found for one-hot encoding")
            return df_encoded
        
        if self.transformers['onehot_encoder'] is None:
            self.transformers['onehot_encoder'] = OneHotEncoder(
                drop='first' if drop_first else None,
                sparse_output=False,
                handle_unknown='ignore'
            )
            
            encoded_array = self.transformers['onehot_encoder'].fit_transform(df_encoded[valid_columns])
            
            save_pickle(self.transformers['onehot_encoder'], 
                       os.path.join(self.models_dir, 'onehot_encoder.pkl'))
        else:
            encoded_array = self.transformers['onehot_encoder'].transform(df_encoded[valid_columns])
        
        feature_names = self.transformers['onehot_encoder'].get_feature_names_out(valid_columns)
        
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_encoded.index)
        
        df_encoded = df_encoded.drop(columns=valid_columns)
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        self.transformed_features.extend(feature_names.tolist())
        
        logger.info(f"One-hot encoded {len(valid_columns)} columns into {len(feature_names)} features")
        return df_encoded
    
    def label_encode_features(self, df, columns=None):
        logger.info("Applying label encoding...")
        
        df_encoded = df.copy()
        col_types = identify_column_types(df_encoded)
        
        if columns is None:
            columns = [col for col in col_types['categorical'] 
                      if df_encoded[col].nunique() > 10]
        
        if not columns:
            logger.warning("No suitable columns found for label encoding")
            return df_encoded
        
        valid_columns = [col for col in columns if col in df_encoded.columns 
                        and col in col_types['categorical']]
        
        for col in valid_columns:
            if col not in self.transformers['label_encoders']:
                self.transformers['label_encoders'][col] = LabelEncoder()
                df_encoded[f'{col}_encoded'] = self.transformers['label_encoders'][col].fit_transform(
                    df_encoded[col].astype(str)
                )
                
                save_pickle(self.transformers['label_encoders'][col],
                           os.path.join(self.models_dir, f'label_encoder_{col}.pkl'))
            else:
                df_encoded[f'{col}_encoded'] = self.transformers['label_encoders'][col].transform(
                    df_encoded[col].astype(str)
                )
            
            self.transformed_features.append(f'{col}_encoded')
        
        logger.info(f"Label encoded {len(valid_columns)} columns")
        return df_encoded
    
    def scale_features(self, df, method='standard', columns=None):
        logger.info(f"Applying {method} scaling...")
        
        df_scaled = df.copy()
        col_types = identify_column_types(df_scaled)
        
        if columns is None:
            columns = col_types['numerical']
        
        valid_columns = [col for col in columns if col in df_scaled.columns 
                        and col in col_types['numerical']]
        
        if not valid_columns:
            logger.warning("No suitable numerical columns found for scaling")
            return df_scaled
        
        if self.transformers['scaler'] is None:
            # Fit new scaler
            if method == 'standard':
                self.transformers['scaler'] = StandardScaler()
            elif method == 'minmax':
                self.transformers['scaler'] = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            df_scaled[valid_columns] = self.transformers['scaler'].fit_transform(df_scaled[valid_columns])
            
            save_pickle(self.transformers['scaler'], 
                       os.path.join(self.models_dir, 'feature_scaler.pkl'))
        else:
            df_scaled[valid_columns] = self.transformers['scaler'].transform(df_scaled[valid_columns])
        
        self.transformed_features.extend(valid_columns)
        
        logger.info(f"Scaled {len(valid_columns)} numerical columns using {method} scaling")
        return df_scaled
    
    def create_polynomial_features(self, df, columns=None, degree=2, max_features=5):
        logger.info("Creating polynomial and interaction features...")

        df_poly = df.copy()
        col_types = identify_column_types(df_poly)
        if columns is None:
            columns = col_types['numerical'][:max_features]
        valid_columns = [col for col in columns if col in df_poly.columns 
                        and col in col_types['numerical']][:max_features]
        
        if len(valid_columns) < 2:
            logger.warning("Need at least 2 numerical columns for polynomial features")
            return df_poly
        
        for col in valid_columns[:5]:
            feature_name = f'{col}_squared'
            df_poly[feature_name] = df_poly[col] ** 2
            self.transformed_features.append(feature_name)
        if len(valid_columns) >= 2:
            for i in range(len(valid_columns)):
                for j in range(i+1, min(i+3, len(valid_columns))):
                        col1, col2 = valid_columns[i], valid_columns[j]
                        feature_name = f'{col1}_x_{col2}'
                        df_poly[feature_name] = df_poly[col1] * df_poly[col2]
                        self.transformed_features.append(feature_name)
        
        logger.info(f"Created polynomial features from {len(valid_columns)} base columns")
        return df_poly
    
    def create_binned_features(self, df, columns=None, n_bins=5, strategy='quantile'):
        logger.info("Creating binned features...")
        
        df_binned = df.copy()
        col_types = identify_column_types(df_binned)
        
        if columns is None:
            columns = [col for col in col_types['numerical'] 
                      if df_binned[col].nunique() > n_bins]
        
        valid_columns = [col for col in columns if col in df_binned.columns 
                        and col in col_types['numerical']]
        
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
                    self.transformed_features.append(feature_name)
                    
                    self.feature_info[feature_name] = {
                        'original_column': col,
                        'bins': bins.tolist(),
                        'strategy': strategy
                    }
                
            except Exception as e:
                logger.warning(f"Could not bin column {col}: {str(e)}")
        
        logger.info(f"Created binned features from {len(valid_columns)} columns")
        return df_binned
    
    def create_derived_features(self, df):
        logger.info("Creating derived features...")
        
        df_derived = df.copy()
        col_types = identify_column_types(df_derived)
        numerical_cols = col_types['numerical']
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
                self.transformed_features.append(feature_name)
            except:
                pass
        if len(numerical_cols) >= 3:
            try:
                feature_name = 'row_mean'
                df_derived[feature_name] = df_derived[numerical_cols].mean(axis=1)
                self.transformed_features.append(feature_name)
                
                feature_name = 'row_std'
                df_derived[feature_name] = df_derived[numerical_cols].std(axis=1)
                self.transformed_features.append(feature_name)
                
                feature_name = 'row_max_min_diff'
                df_derived[feature_name] = (df_derived[numerical_cols].max(axis=1) - 
                                          df_derived[numerical_cols].min(axis=1))
                self.transformed_features.append(feature_name)
            except:
                pass
        categorical_cols = col_types['categorical']
        if len(categorical_cols) >= 2:
            try:
                col1, col2 = categorical_cols[0], categorical_cols[1]
                feature_name = f'{col1}_{col2}_combined'
                df_derived[feature_name] = (df_derived[col1].astype(str) + '_' + 
                                          df_derived[col2].astype(str))
                self.transformed_features.append(feature_name)
            except:
                pass
        
        for col in categorical_cols[:2]:  
            try:
                feature_name = f'{col}_frequency'
                frequency_map = df_derived[col].value_counts().to_dict()
                df_derived[feature_name] = df_derived[col].map(frequency_map)
                self.transformed_features.append(feature_name)
            except:
                pass
        
        logger.info(f"Created derived features, total new features: {len(self.transformed_features)}")
        return df_derived
    
    def engineer_features(self, df, target_col=None, scaling_method='standard', test_size=0.2, random_state=42):
        logger.info("Starting complete feature engineering pipeline...")
        
        df_processed = df.copy()
        print_dataframe_info(df_processed, "Input Dataset")
        
        if target_col and target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
        else:
            logger.warning("Target column not specified or not found. Processing all columns.")
            X = df_processed
            y = None
        
        self.transformed_features = []
        
        X = self.label_encode_features(X)
        
        X = self.one_hot_encode_features(X)
        
        X = self.create_derived_features(X)
        
        X = self.create_binned_features(X)
        X = self.create_polynomial_features(X)
        X = self.scale_features(X, method=scaling_method)
        
        print_dataframe_info(X, "Processed Features")
        
        feature_info = {
            'original_features': df.columns.tolist(),
            'final_features': X.columns.tolist(),
            'transformed_features': self.transformed_features,
            'feature_count': {
                'original': len(df.columns),
                'final': len(X.columns),
                'added': len(self.transformed_features)
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
        
        from utils import save_json
        save_json(feature_info, os.path.join(self.models_dir, 'feature_info.json'))
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if y.dtype == 'object' or y.nunique() <= 10 else None
            )
            
            logger.info(f"Train-test split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            logger.info(f"Feature engineering complete! Created {len(X.columns)} features from {len(df.columns)} original columns")
            
            return X_train, X_test, y_train, y_test
        else:
            logger.info(f"Feature engineering complete! Created {len(X.columns)} features from {len(df.columns)} original columns")
            return X, None, None, None
    
    def transform_new_data(self, df):
        logger.info("Transforming new data using fitted transformers...")
        
        if self.transformers['scaler'] is None:
            try:
                self.transformers['scaler'] = load_pickle(os.path.join(self.models_dir, 'scaler.pkl'))
            except:
                logger.warning("Scaler not found, skipping scaling")
        
        if self.transformers['onehot_encoder'] is None:
            try:
                self.transformers['onehot_encoder'] = load_pickle(os.path.join(self.models_dir, 'onehot_encoder.pkl'))
            except:
                logger.warning("OneHot encoder not found, skipping one-hot encoding")
        
        
        df_transformed = df.copy()
        logger.info("New data transformation complete")
        return df_transformed


def main():
    feature_engineer = FeatureEngineer()
    data_path = "d:/Python/machine-learning-pipeline/data/processed/cleaned_data.csv"
    
    if not os.path.exists(data_path):
        logger.warning(f"Cleaned data not found at {data_path}. Running data cleaning first...")
        from data_cleaning import main as run_cleaning
        run_cleaning()
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    X_train, X_test, y_train, y_test = feature_engineer.engineer_features(
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
    else:
        logger.error("Feature engineering returned None values. Check the engineer_features method.")
    
    print("\nFeature Engineering Complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Created {len(X_train.columns)} features from original dataset")


if __name__ == "__main__":
    main()