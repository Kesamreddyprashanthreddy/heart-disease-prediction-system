import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import setup_logging, identify_column_types

logger = setup_logging()

def plot_correlation_heatmap(df, plots_dir, figsize=(12, 10)):
    logger.info("Creating correlation heatmap...")
    
    col_types = identify_column_types(df)
    numerical_cols = col_types['numerical']
    
    if len(numerical_cols) < 2:
        logger.warning("Need at least 2 numerical columns")
        return pd.DataFrame()
    
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
               mask=mask,
               annot=True, 
               cmap='coolwarm', 
               center=0,
               square=True,
               fmt='.2f',
               cbar_kws={"shrink": .8})
    
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Correlation heatmap saved")
    
    return correlation_matrix

def plot_scatterplots(df, plots_dir, target_col=None, max_features=6):
    logger.info("Creating scatterplot matrix...")
    
    col_types = identify_column_types(df)
    numerical_cols = col_types['numerical']
    
    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if target_col and target_col in df.columns:
        if len(numerical_cols) > max_features:
            correlations = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            selected_cols = correlations.head(max_features).index.tolist()
        else:
            selected_cols = numerical_cols
    else:
        selected_cols = numerical_cols[:max_features]
    
    if len(selected_cols) < 2:
        logger.warning("Need at least 2 numerical columns")
        return
    
    if target_col and target_col in df.columns:
        if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
            g = sns.pairplot(df[selected_cols + [target_col]], hue=target_col, diag_kind='hist')
        else:
            g = sns.pairplot(df[selected_cols], diag_kind='hist')
    else:
        g = sns.pairplot(df[selected_cols], diag_kind='hist')
    
    g.fig.suptitle('Pairwise Scatterplot Matrix', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(plots_dir, 'scatterplot_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Scatterplot matrix saved")
