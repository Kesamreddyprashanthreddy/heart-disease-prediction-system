import pandas as pd
import numpy as np
import os
from utils import (
    setup_logging, 
    ensure_directory, 
    identify_column_types, 
    detect_outliers_iqr,
    print_dataframe_info
)

logger = setup_logging()

class DataCleaner:
    
    def __init__(self):
        self.cleaning_report = {}
        self.outlier_info = {}
    
    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}")
            logger.info(f"Dataset shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df, strategy=None):
        logger.info("Handling missing values...")
        
        df_cleaned = df.copy()
        col_types = identify_column_types(df_cleaned)
        
        missing_before = df_cleaned.isnull().sum().sum()
        
        default_strategy = strategy or {}
        for col in col_types['numerical']:
            if df_cleaned[col].isnull().any():
                if col in default_strategy:
                    if default_strategy[col] == 'mean':
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    elif default_strategy[col] == 'median':
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    elif default_strategy[col] == 'drop':
                        df_cleaned.dropna(subset=[col], inplace=True)
                else:
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                
                logger.info(f"Filled missing values in {col} using median/mean")
        
        for col in col_types['categorical']:
            if df_cleaned[col].isnull().any():
                if col in default_strategy:
                    if default_strategy[col] == 'mode':
                        mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                        df_cleaned[col].fillna(mode_val, inplace=True)
                    elif default_strategy[col] == 'drop':
                        df_cleaned.dropna(subset=[col], inplace=True)
                else:
                    mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                    df_cleaned[col].fillna(mode_val, inplace=True)
                
                logger.info(f"Filled missing values in {col} using mode")
        
        missing_after = df_cleaned.isnull().sum().sum()
        
        self.cleaning_report['missing_values'] = {
            'before': missing_before,
            'after': missing_after,
            'removed': missing_before - missing_after
        }
        
        logger.info(f"Missing values reduced from {missing_before} to {missing_after}")
        return df_cleaned
    
    def remove_duplicates(self, df, keep='first'):
        logger.info("Removing duplicate rows...")
        
        duplicates_before = df.duplicated().sum()
        df_cleaned = df.drop_duplicates(keep=keep)
        duplicates_removed = duplicates_before
        
        self.cleaning_report['duplicates'] = {
            'before': duplicates_before,
            'removed': duplicates_removed,
            'remaining_rows': len(df_cleaned)
        }
        
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        return df_cleaned
    
    def fix_data_types(self, df, type_mapping=None):
        logger.info("Fixing data types...")
        
        df_cleaned = df.copy()
        type_changes = {}
        
        if type_mapping:
            for col, dtype in type_mapping.items():
                if col in df_cleaned.columns:
                    try:
                        old_type = df_cleaned[col].dtype
                        if dtype == 'category':
                            df_cleaned[col] = df_cleaned[col].astype('category')
                        elif dtype == 'datetime':
                            df_cleaned[col] = pd.to_datetime(df_cleaned[col])
                        else:
                            df_cleaned[col] = df_cleaned[col].astype(dtype)
                        
                        type_changes[col] = {'from': str(old_type), 'to': dtype}
                        logger.info(f"Changed {col} from {old_type} to {dtype}")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                try:
                    numeric_col = pd.to_numeric(df_cleaned[col], errors='coerce')
                    if not numeric_col.isnull().all():
                        df_cleaned[col] = numeric_col
                        type_changes[col] = {'from': 'object', 'to': 'numeric'}
                        logger.info(f"Auto-converted {col} to numeric")
                except:
                    pass
        
        self.cleaning_report['type_changes'] = type_changes
        return df_cleaned
    
    def handle_outliers(self, df, method='iqr', action='cap', columns=None):
        logger.info(f"Handling outliers using {method} method with {action} action...")
        
        df_cleaned = df.copy()
        col_types = identify_column_types(df_cleaned)
        
        if columns is None:
            columns = col_types['numerical']
        
        outlier_summary = {}
        
        for col in columns:
            if col in df_cleaned.columns and df_cleaned[col].dtype in ['int64', 'float64']:
                
                if method == 'iqr':
                    outliers, lower_bound, upper_bound = detect_outliers_iqr(df_cleaned, col)
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        if action == 'cap':
                            df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                            df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                            logger.info(f"Capped {outlier_count} outliers in {col}")
                            
                        elif action == 'remove':
                            df_cleaned = df_cleaned[~outliers]
                            logger.info(f"Removed {outlier_count} outlier rows based on {col}")
                            
                        elif action == 'log':
                            if (df_cleaned[col] > 0).all():
                                df_cleaned[col] = np.log1p(df_cleaned[col])
                                logger.info(f"Applied log transformation to {col}")
                        
                        outlier_summary[col] = {
                            'count': outlier_count,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'action': action
                        }
                
                elif method == 'zscore':
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                    outliers = z_scores > 3
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        if action == 'cap':
                            mean_val = df_cleaned[col].mean()
                            std_val = df_cleaned[col].std()
                            lower_bound = mean_val - 3 * std_val
                            upper_bound = mean_val + 3 * std_val
                            
                            df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                            df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                            logger.info(f"Capped {outlier_count} outliers in {col} using z-score")
                        
                        elif action == 'remove':
                            df_cleaned = df_cleaned[~outliers]
                            logger.info(f"Removed {outlier_count} outlier rows based on {col} z-score")
                        
                        outlier_summary[col] = {
                            'count': outlier_count,
                            'method': 'zscore',
                            'action': action
                        }
        
        self.cleaning_report['outliers'] = outlier_summary
        self.outlier_info = outlier_summary
        return df_cleaned
    
    def clean_data(self, filepath, output_path=None, missing_strategy=None,
                   type_mapping=None, outlier_method='iqr', outlier_action='cap'):
        logger.info("Starting complete data cleaning pipeline...")
        
        df = self.load_data(filepath)
        print_dataframe_info(df, "Original Dataset")
        
        df = self.handle_missing_values(df, missing_strategy)
        df = self.remove_duplicates(df)
        df = self.fix_data_types(df, type_mapping)
        df = self.handle_outliers(df, outlier_method, outlier_action)
        
        print_dataframe_info(df, "Cleaned Dataset")
        
        if output_path:
            ensure_directory(os.path.dirname(output_path))
            df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
        
        self.print_cleaning_summary()
        
        return df
    
    def print_cleaning_summary(self):
        print(f"\n{'='*60}")
        print("DATA CLEANING SUMMARY")
        print(f"{'='*60}")
        
        if 'missing_values' in self.cleaning_report:
            mv = self.cleaning_report['missing_values']
            print(f"Missing Values: {mv['before']} → {mv['after']} (removed: {mv['removed']})")
        
        if 'duplicates' in self.cleaning_report:
            dup = self.cleaning_report['duplicates']
            print(f"Duplicates: Removed {dup['removed']} rows")
        
        if 'type_changes' in self.cleaning_report:
            tc = self.cleaning_report['type_changes']
            print(f"Data Type Changes: {len(tc)} columns modified")
            for col, change in tc.items():
                print(f"  - {col}: {change['from']} → {change['to']}")
        
        if 'outliers' in self.cleaning_report:
            outliers = self.cleaning_report['outliers']
            print(f"Outliers Handled: {len(outliers)} columns processed")
            for col, info in outliers.items():
                print(f"  - {col}: {info['count']} outliers {info['action']}ped")


def main():
    cleaner = DataCleaner()
    
    input_path = "d:/Python/machine-learning-pipeline/data/raw/heart_cleveland_upload.csv"
    output_path = "d:/Python/machine-learning-pipeline/data/processed/cleaned_data.csv"
    
    cleaned_df = cleaner.clean_data(
        filepath=input_path,
        output_path=output_path,
        missing_strategy={},
        outlier_method='iqr',
        outlier_action='cap'
    )
    
    print(f"\nCleaning complete! Cleaned data shape: {cleaned_df.shape}")


if __name__ == "__main__":
    main()