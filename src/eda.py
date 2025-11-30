import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional
from utils import (
    setup_logging, 
    ensure_directory, 
    identify_column_types,
    print_dataframe_info
)
logger = setup_logging()

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EDAAnalyzer: 
    def __init__(self, plots_dir: str = "../plots"):
        self.plots_dir = plots_dir
        ensure_directory(plots_dir)
        self.summary_stats = {}
        
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        
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
        
        self.summary_stats = summary
        return summary
    
    def plot_numerical_distributions(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
        
        logger.info("Creating numerical distribution plots...")
        
        col_types = identify_column_types(df)
        numerical_cols = col_types['numerical']
        
        if not numerical_cols:
            logger.warning("No numerical columns found for distribution plots")
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
        plt.savefig(os.path.join(self.plots_dir, 'numerical_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Numerical distribution plots saved")
    
    def plot_boxplots_for_outliers(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 8)) -> None:
        
        logger.info("Creating boxplots for outlier detection...")
        
        col_types = identify_column_types(df)
        numerical_cols = col_types['numerical']
        
        if not numerical_cols:
            logger.warning("No numerical columns found for boxplots")
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
        
        # Hide empty subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'boxplots_outliers.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Boxplot outlier detection plots saved")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> pd.DataFrame:
        logger.info("Creating correlation heatmap...")
        
        col_types = identify_column_types(df)
        numerical_cols = col_types['numerical']
        
        if len(numerical_cols) < 2:
            logger.warning("Need at least 2 numerical columns for correlation heatmap")
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
        plt.savefig(os.path.join(self.plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Correlation heatmap saved")
        
        return correlation_matrix
    
    def plot_scatterplots(self, df: pd.DataFrame, target_col: str = None, max_features: int = 6) -> None:
        
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
            logger.warning("Need at least 2 numerical columns for scatterplots")
            return
        if target_col and target_col in df.columns:
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
                # Categorical target - use for coloring
                g = sns.pairplot(df[selected_cols + [target_col]], hue=target_col, diag_kind='hist')
            else:
              
                g = sns.pairplot(df[selected_cols], diag_kind='hist')
        else:
            g = sns.pairplot(df[selected_cols], diag_kind='hist')
        
        g.fig.suptitle('Pairwise Scatterplot Matrix', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(self.plots_dir, 'scatterplot_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Scatterplot matrix saved")
    
    def plot_categorical_distributions(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
        
        logger.info("Creating categorical distribution plots...")
        
        col_types = identify_column_types(df)
        categorical_cols = col_types['categorical']
        
        if not categorical_cols:
            logger.warning("No categorical columns found for distribution plots")
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
        plt.savefig(os.path.join(self.plots_dir, 'categorical_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Categorical distribution plots saved")
    
    def plot_target_analysis(self, df: pd.DataFrame, target_col: str, figsize: Tuple[int, int] = (15, 6)) -> None:
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in DataFrame")
            return
        
        logger.info(f"Creating target analysis plots for {target_col}...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
   
        if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
            
            value_counts = df[target_col].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(value_counts)))
            
            # Bar plot
            bars = axes[0].bar(range(len(value_counts)), value_counts.values, color=colors, alpha=0.8)
            axes[0].set_title(f'Distribution of {target_col}', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(target_col)
            axes[0].set_ylabel('Count')
            axes[0].set_xticks(range(len(value_counts)))
            axes[0].set_xticklabels(value_counts.index)
            
            # Add value labels
            for bar, value in zip(bars, value_counts.values):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Pie chart
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
        plt.savefig(os.path.join(self.plots_dir, 'target_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Target analysis plots saved")
    
    def generate_eda_report(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        
        logger.info("Generating comprehensive EDA report...")
       
        self.plot_numerical_distributions(df)
        self.plot_boxplots_for_outliers(df)
        correlation_matrix = self.plot_correlation_heatmap(df)
        self.plot_scatterplots(df, target_col)
        self.plot_categorical_distributions(df)
        
        if target_col:
            self.plot_target_analysis(df, target_col)
        
        summary_stats = self.generate_summary_statistics(df)
        
        eda_report = {
            'summary_statistics': summary_stats,
            'correlation_matrix': correlation_matrix.to_dict() if not correlation_matrix.empty else {},
            'plots_generated': [
                'numerical_distributions.png',
                'boxplots_outliers.png',
                'correlation_heatmap.png',
                'scatterplot_matrix.png',
                'categorical_distributions.png'
            ]
        }
        
        if target_col:
            eda_report['plots_generated'].append('target_analysis.png')
        
        from utils import save_json
        save_json(eda_report, os.path.join(self.plots_dir, 'eda_report.json'))
        
        logger.info(f"EDA report complete! {len(eda_report['plots_generated'])} plots generated")
        return eda_report


def main():
    
    eda_analyzer = EDAAnalyzer()
    data_path = "../data/processed/cleaned_data.csv"
    
    if not os.path.exists(data_path):
        logger.warning(f"Cleaned data not found at {data_path}. Running data cleaning first...")
        from data_cleaning import main as run_cleaning
        run_cleaning()
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    report = eda_analyzer.generate_eda_report(df, target_col='loan_approved')
    
    print("\nEDA Analysis Complete!")
    print(f"Generated {len(report['plots_generated'])} visualizations")
    print("Check the 'plots' directory for all generated charts and the EDA report.")


if __name__ == "__main__":
    main()