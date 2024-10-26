import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class CropPricePredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.categorical_columns = ['State', 'Crop', 'Regional_Demand']
        self.numerical_columns = ['CostCultivation', 'Production', 'Yield', 'Temperature', 'RainFall Annual']
        
    def preprocess_data(self, df):
        """Preprocess the data by encoding categorical variables and scaling numerical features"""
        df_processed = df.copy()
        
        # Create copies of encoders for each categorical column
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Select features for the model
        features = self.categorical_columns + self.numerical_columns
        X = df_processed[features].copy()
        y = df_processed['Price']
        
        # Scale the numerical features
        X[self.numerical_columns] = self.scaler.fit_transform(X[self.numerical_columns])
        
        return X, y
    
    def train_model(self, df):
        """Train the Random Forest model"""
        X, y = self.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'feature_importance': self.feature_importance
        }
    
    def predict_price(self, input_data):
        """Predict crop prices based on input features"""
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        input_processed = input_data.copy()
            
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_processed.columns:
                try:
                    input_processed[col] = encoder.transform(input_processed[col])
                except ValueError as e:
                    unique_values = set(self.label_encoders[col].classes_)
                    raise ValueError(f"Invalid value in column '{col}'. Allowed values are: {unique_values}")
        
        # Scale numerical features
        input_processed[self.numerical_columns] = self.scaler.transform(input_processed[self.numerical_columns])
        
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
            # Create test data with different temporal features
            test_data = pd.DataFrame({
                'State': [state] * 3,
                'Crop': [crop] * 3,
                'CostCultivation': [self.scaler.mean_[0]] * 3,  # Using mean values for simulation
                'Production': [self.scaler.mean_[1]] * 3,
                'Yield': [self.scaler.mean_[2]] * 3,
                'Temperature': [self.scaler.mean_[3]] * 3,
                'RainFall Annual': [self.scaler.mean_[4]] * 3,
                'Regional_Demand': ['Medium'] * 3  # Using 'Medium' as default demand
            })
            
            # Get price predictions
            predicted_prices = self.predict_price(test_data)
            avg_predicted_price = np.mean(predicted_prices)
            
            # Generate insights
            price_diff = avg_predicted_price - current_price
            insights['price_trend'] = 'increasing' if price_diff > 0 else 'decreasing'
            
            if price_diff > current_price * 0.1:  # 10% increase expected
                insights['recommendation'] = 'Hold crops for better prices'
            elif price_diff < -current_price * 0.1:  # 10% decrease expected
                insights['recommendation'] = 'Consider selling soon'
            else:
                insights['recommendation'] = 'Market is stable, monitor closely'
                
            # Calculate confidence score based on model's feature importance
            confidence_score = min(0.95, self.model.score(test_data, predicted_prices))
            insights['confidence_score'] = confidence_score
            
        except Exception as e:
            print(f"Error generating market insights: {str(e)}")
            insights['price_trend'] = 'unknown'
            insights['recommendation'] = 'Unable to generate recommendation due to insufficient data'
            insights['confidence_score'] = 0.0
            
        return insights
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = model_data['feature_importance']
        self.categorical_columns = model_data['categorical_columns']
        self.numerical_columns = model_data['numerical_columns']

def main():
    try:
        # Read the data
        df = pd.read_csv('crops_prices_historical_datasets.csv')
        
        # Initialize and train the model
        price_prediction_system = CropPricePredictionSystem()
        evaluation_metrics = price_prediction_system.train_model(df)
        
        print("Model Performance Metrics:")
        print(f"RMSE: {evaluation_metrics['rmse']:.2f}")
        print(f"R2 Score: {evaluation_metrics['r2']:.2f}")
        print("\nFeature Importance:")
        print(evaluation_metrics['feature_importance'])
        
        # Example prediction
        sample_input = pd.DataFrame({
            'State': ['Maharashtra'],
            'Crop': ['WHEAT'],
            'CostCultivation': [15000],
            'Production': [800],
            'Yield': [35],
            'Temperature': [28.5],
            'RainFall Annual': [3000],
            'Regional_Demand': ['Medium']
        })
        
        predicted_price = price_prediction_system.predict_price(sample_input)
        print(f"\nPredicted Price: R{predicted_price[0]:.2f}")
        
        # Get market insights
        insights = price_prediction_system.get_market_insights('WHEAT', 'Maharashtra', predicted_price[0])
        print("\nMarket Insights:")
        print(f"Price Trend: {insights['price_trend']}")
        print(f"Recommendation: {insights['recommendation']}")
        print(f"Confidence Score: {insights['confidence_score']:.2f}")
        
        # Save the model
        price_prediction_system.save_model('crop_price_prediction_model.joblib')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    mani()