import os
import json
import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


def setup_logging(log_file: str = "ml_pipeline.log") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def ensure_directory(directory_path: str) -> None:
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def save_pickle(obj, filepath):
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filepath}")
    return obj


def save_json(data: Dict, filepath: str) -> None:
    import numpy as np
    
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    ensure_directory(os.path.dirname(filepath))
    converted_data = convert_numpy_types(data)
    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=4)
    print(f"JSON data saved to {filepath}")


def load_json(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"JSON data loaded from {filepath}")
    return data


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


def get_memory_usage(df: pd.DataFrame) -> str:
    memory_usage = df.memory_usage(deep=True).sum()
    if memory_usage < 1024:
        return f"{memory_usage} bytes"
    elif memory_usage < 1024**2:
        return f"{memory_usage/1024:.2f} KB"
    elif memory_usage < 1024**3:
        return f"{memory_usage/(1024**2):.2f} MB"
    else:
        return f"{memory_usage/(1024**3):.2f} GB"


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Tuple[np.ndarray, float, float]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers, lower_bound, upper_bound


def print_dataframe_info(df: pd.DataFrame, title: str = "DataFrame Info") -> None:
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {get_memory_usage(df)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    col_types = identify_column_types(df)
    print(f"Numerical columns ({len(col_types['numerical'])}): {col_types['numerical']}")
    print(f"Categorical columns ({len(col_types['categorical'])}): {col_types['categorical']}")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])


def validate_target_column(df: pd.DataFrame, target_col: str) -> str:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    target = df[target_col]
    
    if pd.api.types.is_numeric_dtype(target):
        unique_values = target.nunique()
        if unique_values <= 10 and target.dtype in ['int64', 'int32']:
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'


def create_sample_dataset(filepath: str, n_samples: int = 500) -> None:
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'married': np.random.choice(['Yes', 'No'], n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.normal(200000, 100000, n_samples),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df['loan_approved'] = (
        (df['income'] > 40000) & 
        (df['credit_score'] > 650) & 
        (df['employment_type'].isin(['Full-time', 'Self-employed']))
    ).astype(int)
    missing_cols = ['income', 'credit_score', 'education_years']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    duplicate_rows = df.sample(n=int(0.02 * len(df)))
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    ensure_directory(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)
    print(f"Sample dataset created with {len(df)} rows and saved to {filepath}")


if __name__ == "__main__":
    sample_path = "../data/raw/sample_data.csv"
    create_sample_dataset(sample_path)