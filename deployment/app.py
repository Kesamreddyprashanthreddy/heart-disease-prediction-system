from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_pickle, load_json, identify_column_types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

best_model = None
feature_info = None
scaler = None
label_encoders = {}
onehot_encoder = None
problem_type = None
model_metadata = None

def load_model_artifacts():
    global best_model, feature_info, scaler, label_encoders, onehot_encoder, problem_type, model_metadata
    
    models_dir = "d:/Python/machine-learning-pipeline/models"
    
    try:
        best_model_path = os.path.join(models_dir, 'best_model.pkl')
        if os.path.exists(best_model_path):
            best_model = load_pickle(best_model_path)
            logger.info("Best model loaded successfully")
        else:
            logger.error("Best model not found")
            return False
        
        feature_info_path = os.path.join(models_dir, 'feature_info.json')
        if os.path.exists(feature_info_path):
            feature_info = load_json(feature_info_path)
            logger.info("Feature info loaded successfully")
        
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = load_pickle(scaler_path)
            logger.info("Scaler loaded successfully")
        
        # Load one-hot encoder
        onehot_path = os.path.join(models_dir, 'onehot_encoder.pkl')
        if os.path.exists(onehot_path):
            onehot_encoder = load_pickle(onehot_path)
            logger.info("One-hot encoder loaded successfully")
        
        for file in os.listdir(models_dir):
            if file.startswith('label_encoder_') and file.endswith('.pkl'):
                column_name = file.replace('label_encoder_', '').replace('.pkl', '')
                label_encoders[column_name] = load_pickle(os.path.join(models_dir, file))
        
        if label_encoders:
            logger.info(f"Loaded {len(label_encoders)} label encoders")
        
        best_model_info_path = os.path.join(models_dir, 'best_model_info.json')
        if os.path.exists(best_model_info_path):
            model_metadata = load_json(best_model_info_path)
            if 'classification' in model_metadata:
                problem_type = 'classification'
            elif 'regression' in model_metadata:
                problem_type = 'regression'
            logger.info(f"Problem type detected: {problem_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return False


def validate_input_data(data: Dict[str, Any]) -> Tuple[bool, str, pd.DataFrame]:
    try:
        
        if isinstance(data, dict):
            
            df = pd.DataFrame([data])
        elif isinstance(data, list):
           
            df = pd.DataFrame(data)
        else:
            return False, "Input must be a dictionary or list of dictionaries", None
        
        if df.empty:
            return False, "Input data is empty", None
        
        if feature_info and 'original_features' in feature_info:
            expected_features = [f for f in feature_info['original_features'] if f != 'loan_approved']  # Remove target column
            
           
            missing_features = set(expected_features) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                
                for feature in missing_features:
                    df[feature] = 0  
        
        return True, "", df
        
    except Exception as e:
        return False, f"Data validation error: {str(e)}", None


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_processed = df.copy()
        
        col_types = identify_column_types(df_processed)
        
        for col, encoder in label_encoders.items():
            if col in df_processed.columns:
               
                try:
                    df_processed[f'{col}_encoded'] = encoder.transform(df_processed[col].astype(str))
                except ValueError:
                    
                    df_processed[f'{col}_encoded'] = 0
        
        if onehot_encoder is not None:
            categorical_cols = col_types['categorical']
         
            available_cats = [col for col in categorical_cols if col in df_processed.columns]
            
            if available_cats:
                try:
                    encoded_array = onehot_encoder.transform(df_processed[available_cats])
                    feature_names = onehot_encoder.get_feature_names_out(available_cats)
                    
                    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_processed.index)
                    df_processed = df_processed.drop(columns=available_cats)
                    df_processed = pd.concat([df_processed, encoded_df], axis=1)
                except:
                    logger.warning("Error in one-hot encoding, skipping")

        numerical_cols = col_types['numerical']
        if len(numerical_cols) >= 2:
            df_processed['row_mean'] = df_processed[numerical_cols].mean(axis=1)
            
            if len(numerical_cols) >= 2:
                col1, col2 = numerical_cols[0], numerical_cols[1]
                df_processed[f'{col1}_x_{col2}'] = df_processed[col1] * df_processed[col2]
                df_processed[f'{col1}_squared'] = df_processed[col1] ** 2
        
        if scaler is not None and numerical_cols:
            scalable_cols = [col for col in numerical_cols if col in df_processed.columns]
            if scalable_cols:
                try:
                    df_processed[scalable_cols] = scaler.transform(df_processed[scalable_cols])
                except:
                    logger.warning("Error in scaling, skipping")
        
        if feature_info and 'final_features' in feature_info:
            expected_features = feature_info['final_features']
            
            for feature in expected_features:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0
            
            df_processed = df_processed.reindex(columns=expected_features, fill_value=0)
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        logger.error(traceback.format_exc())
        return df


@app.route('/', methods=['GET'])
def index():
   
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': best_model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    if best_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_type': str(type(best_model).__name__),
        'problem_type': problem_type,
        'model_metadata': model_metadata,
        'features_available': feature_info.get('final_features', []) if feature_info else [],
        'feature_count': len(feature_info.get('final_features', [])) if feature_info else 0
    }
    
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expected JSON format:
    {
        "data": {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    }
    
    Or for batch predictions:
    {
        "data": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value3, "feature2": value4, ...}
        ]
    }
    """
    try:
        if best_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing data field in request'}), 400
        
        is_valid, error_msg, df = validate_input_data(request_data['data'])
        
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        df_processed = preprocess_input(df)
        
        predictions = best_model.predict(df_processed)
        probabilities = None
        if problem_type == 'classification' and hasattr(best_model, 'predict_proba'):
            try:
                probabilities = best_model.predict_proba(df_processed)
            except:
                probabilities = None
        
        response = {
            'predictions': predictions.tolist(),
            'prediction_count': len(predictions),
            'model_info': {
                'model_type': str(type(best_model).__name__),
                'problem_type': problem_type
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()
        
        if len(predictions) == 1:
            response['single_prediction'] = {
                'value': float(predictions[0]),
                'interpretation': get_prediction_interpretation(predictions[0], probabilities[0] if probabilities is not None else None)
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


def get_prediction_interpretation(prediction: float, probabilities: np.ndarray = None) -> str:
    
    if problem_type == 'classification':
        if probabilities is not None and len(probabilities) == 2:
            confidence = max(probabilities) * 100
            result = "Approved" if prediction == 1 else "Rejected"
            return f"{result} with {confidence:.1f}% confidence"
        else:
            return f"Class: {int(prediction)}"
    else:
        return f"Predicted value: {prediction:.2f}"


@app.route('/predict_sample', methods=['GET'])
def predict_sample():
    
    if best_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    sample_data = {
        'age': 35,
        'income': 50000,
        'education_years': 16,
        'experience': 10,
        'city': 'New York',
        'gender': 'Male',
        'married': 'Yes',
        'credit_score': 720,
        'loan_amount': 200000,
        'employment_type': 'Full-time'
    }
    
    try:
        is_valid, error_msg, df = validate_input_data(sample_data)
        
        if not is_valid:
            return jsonify({'error': f'Sample validation failed: {error_msg}'}), 500
        
        df_processed = preprocess_input(df)
        prediction = best_model.predict(df_processed)[0]
        
        probabilities = None
        if problem_type == 'classification' and hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(df_processed)[0]
        
        response = {
            'sample_input': sample_data,
            'prediction': float(prediction),
            'interpretation': get_prediction_interpretation(prediction, probabilities),
            'model_type': str(type(best_model).__name__),
            'timestamp': datetime.now().isoformat()
        }
        
        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Sample prediction error: {str(e)}")
        return jsonify({'error': f'Sample prediction failed: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Loading model artifacts...")
    
    if load_model_artifacts():
        logger.info("Model artifacts loaded successfully")
        logger.info(f"Starting Flask API server...")
        logger.info(f"Available endpoints:")
        logger.info(f"  GET  /health - Health check")
        logger.info(f"  GET  /model_info - Model information")
        logger.info(f"  POST /predict - Main prediction endpoint")
        logger.info(f"  GET  /predict_sample - Sample prediction")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load model artifacts. Please ensure models are trained.")
        sys.exit(1)