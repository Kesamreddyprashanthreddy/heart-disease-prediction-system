import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import setup_logging, ensure_directory, identify_column_types

logger = setup_logging()

def plot_numerical_distributions(df, plots_dir, figsize=(15, 10)):
    logger.info("Creating numerical distribution plots...")
    
    col_types = identify_column_types(df)
    numerical_cols = col_types['numerical']
    
    if not numerical_cols:
        logger.warning("No numerical columns found")
        return
    
    n_cols = min(4, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes if len(numerical_cols) > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i] if len(numerical_cols) > 1 else axes
        
        ax.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'numerical_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Numerical distribution plots saved")

def plot_boxplots_for_outliers(df, plots_dir, figsize=(15, 8)):
    logger.info("Creating boxplots for outlier detection...")
    
    col_types = identify_column_types(df)
    numerical_cols = col_types['numerical']
    
    if not numerical_cols:
        logger.warning("No numerical columns found")
        return
    
    n_cols = min(4, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes if len(numerical_cols) > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        ax = axes[i] if len(numerical_cols) > 1 else axes
        
        box_plot = ax.boxplot(df[col].dropna(), patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][0].set_alpha(0.7)
        
        ax.set_title(f'Boxplot: {col}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        outlier_text = f'Outliers: {len(outliers)}\nIQR: {IQR:.2f}'
        ax.text(0.02, 0.98, outlier_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplots_outliers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Boxplot outlier detection plots saved")
