import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CropPricePredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = ['State', 'Crop', 'Regional_Demand']
        self.numerical_columns = ['CostCultivation', 'Production', 'Yield', 'Temperature', 'RainFall Annual']
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data by encoding categorical variables and scaling numerical features"""
        df_processed = df.copy()

        # Label encode categorical features
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            if is_training:
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])

        # Scale numerical features
        if is_training:
            df_processed[self.numerical_columns] = self.scaler.fit_transform(df_processed[self.numerical_columns])
        else:
            df_processed[self.numerical_columns] = self.scaler.transform(df_processed[self.numerical_columns])

        return df_processed[self.categorical_columns + self.numerical_columns], df_processed['Price']
    
    def train_model(self, df):
        """Train the Random Forest model"""
        X, y = self.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logging.info("Model training complete. MSE: %.2f, R2: %.2f", mse, r2)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
    
    def predict_price(self, input_data):
        """Predict crop prices based on input features"""
        input_processed = input_data.copy()

        # Encode categorical variables
        for col in self.categorical_columns:
            input_processed[col] = self.label_encoders[col].transform(input_processed[col])

        # Scale numerical features
        input_processed[self.numerical_columns] = self.scaler.transform(input_processed[self.numerical_columns])

        # Ensure the column order matches training
        input_processed = input_processed[self.categorical_columns + self.numerical_columns]

        # Make prediction
        prediction = self.model.predict(input_processed)
        return prediction
    
    def get_market_insights(self, crop, state, current_price):
        """Generate market insights and recommendations"""
        insights = {
            'price_trend': None,
            'recommendation': None,
            'confidence_score': None
        }
        
        try:
            # Create test data with hypothetical values
            test_data = pd.DataFrame({
                'State': [state] * 3,
                'Crop': [crop] * 3,
                'CostCultivation': [self.scaler.mean_[0]] * 3,
                'Production': [self.scaler.mean_[1]] * 3,
                'Yield': [self.scaler.mean_[2]] * 3,
                'Temperature': [self.scaler.mean_[3]] * 3,
                'RainFall Annual': [self.scaler.mean_[4]] * 3,
                'Regional_Demand': ['Medium'] * 3
            })
            
            predicted_prices = self.predict_price(test_data)
            avg_predicted_price = np.mean(predicted_prices)
            
            price_diff = avg_predicted_price - current_price
            insights['price_trend'] = 'increasing' if price_diff > 0 else 'decreasing'
            
            if price_diff > current_price * 0.1:
                insights['recommendation'] = 'Hold crops for better prices'
            elif price_diff < -current_price * 0.1:
                insights['recommendation'] = 'Consider selling soon'
            else:
                insights['recommendation'] = 'Market is stable, monitor closely'
                
            confidence_score = min(0.95, self.model.score(test_data, predicted_prices))
            insights['confidence_score'] = confidence_score
            
        except Exception as e:
            logging.error(f"Error generating market insights: {str(e)}")
            insights.update({'price_trend': 'unknown', 'recommendation': 'Unable to generate', 'confidence_score': 0.0})
            
        return insights
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.categorical_columns = model_data['categorical_columns']
        self.numerical_columns = model_data['numerical_columns']

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load or train model
price_prediction_system = CropPricePredictionSystem()
try:
    price_prediction_system.load_model('crop_price_prediction_model.joblib')
    logging.info("Model loaded successfully")
except Exception:
    logging.warning("Model not found, training a new model")
    df = pd.read_csv('crops_prices_historical_datasets.csv')
    price_prediction_system.train_model(df)
    price_prediction_system.save_model('crop_price_prediction_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame({
            'State': [data['Region']],
            'Crop': [data.get('Crop', 'WHEAT')],
            'CostCultivation': [float(data['CostCultivation'])],
            'Production': [float(data['Production'])],
            'Yield': [float(data['Yield'])],
            'Temperature': [float(data['Temperature'])],
            'RainFall Annual': [float(data['RainFallAnnual'])],
            'Regional_Demand': [data.get('RegionalDemand', 'Medium')]
        })
        
        predicted_price = price_prediction_system.predict_price(input_data)
        insights = price_prediction_system.get_market_insights(data.get('Crop', 'WHEAT'), data['Region'], predicted_price[0])
        
        return jsonify({
            'predictedPrice': float(predicted_price[0]),
            'priceTrend': insights['price_trend'],
            'recommendation': insights['recommendation'],
            'confidenceScore': float(insights['confidence_score'])
        })
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
