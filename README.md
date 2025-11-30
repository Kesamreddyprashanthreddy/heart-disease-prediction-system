# Machine Learning Pipeline Project

A complete end-to-end machine learning pipeline with data cleaning, feature engineering, model training, evaluation, and deployment.

## Project Structure

```
machine-learning-pipeline/
│── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data files
│── notebooks/
│   └── ml_pipeline_complete.ipynb  # Complete pipeline notebook
│── src/
│   ├── data_cleaning.py        # Data cleaning functions
│   ├── feature_engineering.py  # Feature engineering functions
│   ├── model_training.py       # Model training functions
│   ├── model_evaluation.py     # Model evaluation functions
│   └── utils.py               # Utility functions
│── models/                     # Trained models and metadata
│── deployment/
│   ├── app.py                 # Flask API application
│   └── test_api.py            # API testing script
│── plots/                      # Generated visualizations
│── docs/                       # Documentation
│── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

- **Data Cleaning**: Handle missing values, duplicates, outliers using IQR method
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical summaries
- **Feature Engineering**: One-hot encoding, label encoding, standardization, feature creation, binning
- **Model Training**: Multiple algorithms (Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics, confusion matrices, ROC curves, comparison tables
- **Deployment**: Flask API with input validation for real-time predictions

## Supported Models

### Classification

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Regression

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

Option A - Using Jupyter Notebook (Recommended):

```bash
jupyter notebook notebooks/ml_pipeline_complete.ipynb
```

Option B - Using Individual Modules:

```bash
cd src
python data_cleaning.py
python feature_engineering.py
python model_training.py
python model_evaluation.py
```

### 3. Start the API Server

```bash
cd deployment
python app.py
```

The API will be available at `http://localhost:5000`

### 4. Test the API

```bash
cd deployment
python test_api.py
```

## API Documentation

### Endpoints

- **GET** `/health` - Health check and server status
- **GET** `/model_info` - Information about the loaded model
- **POST** `/predict` - Make predictions on new data
- **GET** `/predict_sample` - Test prediction with sample data

### Example Usage

#### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "age": 35,
      "income": 55000,
      "education_years": 16,
      "experience": 10,
      "city": "New York",
      "gender": "Female",
      "married": "Yes",
      "credit_score": 720,
      "loan_amount": 200000,
      "employment_type": "Full-time"
    }
  }'
```

#### Batch Predictions

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"age": 25, "income": 35000, "credit_score": 600, ...},
      {"age": 45, "income": 80000, "credit_score": 780, ...}
    ]
  }'
```

## Data Requirements

The pipeline is designed to work with any CSV dataset that contains:

- **Mixed data types**: Both numerical and categorical features
- **Minimum size**: 300+ rows recommended for robust model training
- **Target variable**: Required for supervised learning
- **Feature variety**: The more diverse features, the better the model performance

### Sample Dataset Features

The included sample dataset demonstrates a loan approval prediction scenario with features like:

- `age`, `income`, `education_years` (numerical)
- `city`, `gender`, `employment_type` (categorical)
- `loan_approved` (target variable)

## Pipeline Components

### 1. Data Cleaning (`src/data_cleaning.py`)

- Missing value imputation (mean/median/mode)
- Duplicate removal
- Data type correction
- Outlier detection and handling using IQR method
- Comprehensive data quality reporting

### 2. Exploratory Data Analysis (`src/eda.py`)

- Statistical summaries for all features
- Distribution plots (histograms, boxplots)
- Correlation analysis with heatmaps
- Feature relationship visualizations
- Target variable analysis

### 3. Feature Engineering (`src/feature_engineering.py`)

- One-hot encoding for categorical variables
- Label encoding for high-cardinality features
- Feature scaling (StandardScaler/MinMaxScaler)
- Polynomial and interaction features
- Binning for continuous variables
- Custom derived features

### 4. Model Training (`src/model_training.py`)

- Multiple algorithm support
- Automated hyperparameter tuning with GridSearchCV
- Cross-validation for model selection
- Model persistence with pickle
- Training history and metadata tracking

### 5. Model Evaluation (`src/model_evaluation.py`)

- Classification metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Regression metrics: RMSE, MAE, R², MAPE
- Confusion matrices and ROC curves
- Model comparison visualizations
- Best model selection and saving

### 6. Deployment (`deployment/app.py`)

- Flask REST API with input validation
- Model loading and preprocessing pipeline
- Error handling and logging
- Batch and single prediction support
- API documentation and health checks

## Customization

### Using Your Own Dataset

1. Replace `data/raw/sample_data.csv` with your dataset
2. Update the target column name in the pipeline scripts
3. Adjust feature engineering based on your data characteristics
4. Run the pipeline: `jupyter notebook notebooks/ml_pipeline_complete.ipynb`

### Adding New Models

1. Edit `src/model_training.py`
2. Add your model to the `setup_models()` function
3. Define hyperparameters for tuning
4. Re-run the training pipeline

### Custom Feature Engineering

1. Edit `src/feature_engineering.py`
2. Add new transformation functions
3. Update the `engineer_features()` method
4. Test with your dataset

## Output Files

After running the pipeline, the following files will be generated:

### Data Files (`data/`)

- `raw/sample_data.csv` - Original dataset
- `processed/cleaned_data.csv` - Cleaned dataset
- `processed/X_train.csv`, `X_test.csv` - Feature matrices
- `processed/y_train.csv`, `y_test.csv` - Target vectors

### Model Files (`models/`)

- `best_model.pkl` - Best performing model
- `scaler.pkl` - Feature scaler
- `onehot_encoder.pkl` - Categorical encoder
- `label_encoder_*.pkl` - Label encoders for each categorical column
- `*.pkl` - Individual trained models
- `*.json` - Model metadata and evaluation results

### Visualization Files (`plots/`)

- `numerical_distributions.png` - Feature distributions
- `correlation_heatmap.png` - Feature correlations
- `confusion_matrices.png` - Model performance
- `roc_curves.png` - ROC analysis
- `eda_report.json` - Complete EDA summary

## Best Practices

1. **Data Quality**: Always inspect and clean your data before modeling
2. **Feature Engineering**: Domain knowledge helps create meaningful features
3. **Model Selection**: Use cross-validation to avoid overfitting
4. **Evaluation**: Use appropriate metrics for your problem type
5. **Deployment**: Test your API thoroughly before production use
6. **Monitoring**: Log predictions and monitor model performance over time

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `requirements.txt`
2. **Memory Issues**: For large datasets, consider sampling or incremental processing
3. **API Errors**: Check that the model files exist in the `models/` directory
4. **Performance Issues**: Reduce hyperparameter grid size for faster training

### Getting Help

1. Check the Jupyter notebook for detailed examples
2. Review the API test script for usage patterns
3. Examine log files for detailed error messages
4. Ensure your data follows the expected format

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with scikit-learn, XGBoost, and Flask
- Inspired by MLOps best practices
- Designed for educational and production use

---

**Ready to build amazing ML pipelines? Start with the Jupyter notebook and explore the possibilities!** 🚀
#   h e a r t - d i s e a s e - p r e d i c t i o n - s y s t e m  
 