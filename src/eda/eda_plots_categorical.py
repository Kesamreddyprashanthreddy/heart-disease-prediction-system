import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import setup_logging, identify_column_types

logger = setup_logging()

def plot_categorical_distributions(df, plots_dir, figsize=(15, 10)):
    logger.info("Creating categorical distribution plots...")
    
    col_types = identify_column_types(df)
    categorical_cols = col_types['categorical']
    
    if not categorical_cols:
        logger.warning("No categorical columns found")
        return
    
    n_cols = min(3, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes if len(categorical_cols) > 1 else [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    for i, col in enumerate(categorical_cols):
        ax = axes[i] if len(categorical_cols) > 1 else axes
        
        value_counts = df[col].value_counts()
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                     color=colors[i % len(colors)], alpha=0.8, edgecolor='black')
        
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, value_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        percentages = (value_counts / len(df) * 100).round(1)
        stats_text = f'Unique: {len(value_counts)}\nMost common: {value_counts.index[0]} ({percentages.iloc[0]}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'categorical_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Categorical distribution plots saved")

def plot_target_analysis(df, plots_dir, target_col, figsize=(15, 6)):
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return
    
    logger.info(f"Creating target analysis plots for {target_col}...")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
        value_counts = df[target_col].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(value_counts)))
        bars = axes[0].bar(range(len(value_counts)), value_counts.values, color=colors, alpha=0.8)
        axes[0].set_title(f'Distribution of {target_col}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(value_counts)))
        axes[0].set_xticklabels(value_counts.index)
        for bar, value in zip(bars, value_counts.values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
        
        axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title(f'Proportion of {target_col}', fontsize=12, fontweight='bold')
        
    else:
        axes[0].hist(df[target_col].dropna(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0].set_title(f'Distribution of {target_col}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        box_plot = axes[1].boxplot(df[target_col].dropna(), patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        axes[1].set_title(f'Boxplot of {target_col}', fontsize=12, fontweight='bold')
        axes[1].set_ylabel(target_col)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'target_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Target analysis plots saved")
