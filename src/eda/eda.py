import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import setup_logging, ensure_directory
from eda_stats import generate_summary_statistics
from eda_plots_numerical import plot_numerical_distributions, plot_boxplots_for_outliers
from eda_plots_correlation import plot_correlation_heatmap, plot_scatterplots
from eda_plots_categorical import plot_categorical_distributions, plot_target_analysis

logger = setup_logging()
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_eda_report(df, plots_dir, target_col=None):
    logger.info("Generating comprehensive EDA report...")
    
    ensure_directory(plots_dir)
    
    plot_numerical_distributions(df, plots_dir)
    plot_boxplots_for_outliers(df, plots_dir)
    correlation_matrix = plot_correlation_heatmap(df, plots_dir)
    plot_scatterplots(df, plots_dir, target_col)
    plot_categorical_distributions(df, plots_dir)
    
    if target_col:
        plot_target_analysis(df, plots_dir, target_col)
    
    summary_stats = generate_summary_statistics(df)
    
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
    save_json(eda_report, os.path.join(plots_dir, 'eda_report.json'))
    
    logger.info(f"EDA report complete! {len(eda_report['plots_generated'])} plots generated")
    return eda_report

def main():
    # Run EDA analysis
    plots_dir = "../plots"
    data_path = "../data/processed/cleaned_data.csv"
    
    if not os.path.exists(data_path):
        logger.warning(f"Cleaned data not found at {data_path}")
        from data_cleaning import main as run_cleaning
        run_cleaning()
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    report = generate_eda_report(df, plots_dir, target_col='loan_approved')
    
    print("\nEDA Analysis Complete!")
    print(f"Generated {len(report['plots_generated'])} visualizations")
    print("Check the 'plots' directory for all generated charts.")

if __name__ == "__main__":
    main()