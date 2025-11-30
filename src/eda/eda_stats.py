import pandas as pd
from utils import setup_logging, identify_column_types

logger = setup_logging()

def generate_summary_statistics(df):
    logger.info("Generating summary statistics...")
    
    col_types = identify_column_types(df)
    
    summary = {
        'dataset_info': {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum()
        },
        'numerical_summary': {},
        'categorical_summary': {}
    }
    
    if col_types['numerical']:
        numerical_stats = df[col_types['numerical']].describe()
        summary['numerical_summary'] = numerical_stats.to_dict()
        
        for col in col_types['numerical']:
            summary['numerical_summary'][col].update({
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'variance': df[col].var()
            })
    
    if col_types['categorical']:
        cat_summary = {}
        for col in col_types['categorical']:
            cat_summary[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency_most_common': df[col].value_counts().iloc[0] if not df[col].empty else 0,
                'missing_count': df[col].isnull().sum()
            }
        summary['categorical_summary'] = cat_summary
    
    return summary
