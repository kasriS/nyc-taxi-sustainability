import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class TaxiDurationPredictor:
    def __init__(self, model_dir='results/models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.selected_features = None
        self.models = {}
        self.stacked_model = None
        self.feature_importance = {}
        
        # CO2 emission factors (kg CO2 per km)
        self.co2_factors = {
            'taxi': 0.21,  # Average taxi
            'electric_taxi': 0.05,  # Electric taxi
            'public_transport': 0.089,  # Average public transport
            'private_car': 0.171,  # Average private car
            'optimized_route': 0.18  # 15% reduction from route optimization
        }
        
    def prepare_features(self, df, target_col='trip_duration'):
        """Prepare features for modeling"""
        # Exclude non-feature columns
        exclude_cols = [
            'id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'date', 
            'pickup_time_bin', 'pickup_geohash', 'pickup_borough', 'dropoff_borough'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        # Prepare target if available
        y = None
        if target_col in df.columns:
            y = np.log1p(df[target_col])  # Log transform for trip duration
        
        return X, y, feature_cols
    
    def select_features(self, X, y, method='combined'):
        """Feature selection using multiple methods"""
        print("Starting feature selection...")
        
        # Scale features for linear model
        X_scaled = self.scaler.fit_transform(X)
        
        # Method 1: Lasso CV for linear feature selection
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        linear_features = X.columns[lasso.coef_ != 0]
        print(f"Lasso selected {len(linear_features)} features")
        
        # Method 2: Tree-based feature selection
        xgb_selector = xgb.XGBRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=6,
            eval_metric='rmse'
        )
        xgb_selector.fit(X_scaled, y)
        
        selector = SelectFromModel(xgb_selector, prefit=True, threshold="median")
        tree_features = X.columns[selector.get_support()]
        print(f"XGBoost selected {len(tree_features)} features")
        
        # Combine selected features
        if method == 'combined':
            self.selected_features = list(set(linear_features).union(set(tree_features)))
        elif method == 'lasso':
            self.selected_features = list(linear_features)
        elif method == 'tree':
            self.selected_features = list(tree_features)
        
        print(f"Final selected features: {len(self.selected_features)}")
        return self.selected_features
    
    def train_individual_models(self, X, y):
        """Train individual models"""
        print("Training individual models...")
        
        # XGBoost
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
        # LightGBM
        self.models['lgb'] = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            n_jobs=-1,
            verbose=-1
        )
        
        # CatBoost
        self.models['cat'] = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Random Forest
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Train all models
        cv_scores = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Fit the model
            if name in ['xgb', 'lgb']:
                # Use validation for early stopping
                X_train, X_val = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
                y_train, y_val = y[:int(0.8*len(y))], y[int(0.8*len(y)):]
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X, y)
            
            # Cross-validation score
            cv_score = cross_val_score(
                model, X, y, cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            ).mean()
            cv_scores[name] = -cv_score
            print(f"{name} CV RMSE: {np.sqrt(-cv_score):.4f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    self.selected_features, model.feature_importances_
                ))
        
        return cv_scores
    
    def train_stacked_model(self, X, y):
        """Train stacked ensemble model"""
        print("Training stacked model...")
        
        # Create base models for stacking
        base_models = [
            ('xgb', self.models['xgb']),
            ('lgb', self.models['lgb']),
            ('cat', self.models['cat'])
        ]
        
        # Meta-learner
        meta_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Stacked model
        self.stacked_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1
        )
        
        # Train stacked model
        self.stacked_model.fit(X, y)
        
        # Evaluate stacked model
        cv_score = cross_val_score(
            self.stacked_model, X, y, cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        ).mean()
        
        print(f"Stacked Model CV RMSE: {np.sqrt(-cv_score):.4f}")
        return np.sqrt(-cv_score)
    
    def train_pipeline(self, df, target_col='trip_duration'):
        """Complete training pipeline"""
        print("Starting model training pipeline...")
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df, target_col)
        
        # Feature selection
        selected_features = self.select_features(X, y)
        X_selected = X[selected_features]
        
        # Scale selected features
        X_scaled = self.scaler.fit_transform(X_selected)
        X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
        
        # Train individual models
        cv_scores = self.train_individual_models(X_scaled, y)
        
        # Train stacked model
        stacked_score = self.train_stacked_model(X_scaled, y)
        
        # Save models
        self.save_models()
        
        return {
            'individual_scores': cv_scores,
            'stacked_score': stacked_score,
            'selected_features': selected_features
        }
    
    def predict(self, df, use_stacked=True):
        """Make predictions on new data"""
        # Prepare features
        X, _, _ = self.prepare_features(df)
        
        # Select features
        if self.selected_features:
            X_selected = X[self.selected_features]
        else:
            X_selected = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns)
        
        # Make predictions
        if use_stacked and self.stacked_model:
            log_predictions = self.stacked_model.predict(X_scaled)
        else:
            # Use ensemble of individual models
            predictions = []
            for name, model in self.models.items():
                if name in ['xgb', 'lgb', 'cat']:  # Use best models
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
            
            log_predictions = np.mean(predictions, axis=0)
        
        # Transform back from log scale
        predictions = np.expm1(log_predictions)
        
        # Clip predictions to reasonable range
        predictions = np.clip(predictions, 30, 7200)  # 30 seconds to 2 hours
        
        return predictions
    
    def calculate_co2_emissions(self, distance_km, transport_mode='taxi'):
        """Calculate CO2 emissions for different transport modes"""
        if transport_mode not in self.co2_factors:
            transport_mode = 'taxi'
        
        return distance_km * self.co2_factors[transport_mode]
    
    def estimate_sustainability_impact(self, df, predictions):
        """Estimate sustainability impact of optimized routing"""
        results = {}
        
        # Calculate baseline emissions (current taxi)
        if 'haversine_dist' in df.columns:
            distances = df['haversine_dist'].values
        elif 'total_distance' in df.columns:
            distances = df['total_distance'].values / 1000  # Convert to km
        else:
            # Estimate distance from coordinates
            distances = np.sqrt(
                (df['pickup_latitude'] - df['dropoff_latitude'])**2 +
                (df['pickup_longitude'] - df['dropoff_longitude'])**2
            ) * 111  # Rough conversion to km
        
        # Current emissions
        current_emissions = self.calculate_co2_emissions(distances, 'taxi')
        
        # Optimized routing emissions (15% reduction)
        optimized_emissions = self.calculate_co2_emissions(distances, 'optimized_route')
        
        # Alternative transport modes
        electric_taxi_emissions = self.calculate_co2_emissions(distances, 'electric_taxi')
        public_transport_emissions = self.calculate_co2_emissions(distances, 'public_transport')
        
        # Calculate savings
        results = {
            'total_trips': len(df),
            'total_distance_km': distances.sum(),
            'current_co2_kg': current_emissions.sum(),
            'optimized_co2_kg': optimized_emissions.sum(),
            'electric_taxi_co2_kg': electric_taxi_emissions.sum(),
            'public_transport_co2_kg': public_transport_emissions.sum(),
            'co2_reduction_kg': current_emissions.sum() - optimized_emissions.sum(),
            'co2_reduction_percent': (current_emissions.sum() - optimized_emissions.sum()) / current_emissions.sum() * 100,
            'avg_trip_distance_km': distances.mean(),
            'avg_current_co2_per_trip_kg': current_emissions.mean(),
            'avg_optimized_co2_per_trip_kg': optimized_emissions.mean()
        }
        
        return results
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        # Convert from log scale
        y_true_exp = np.expm1(y_true) if y_true.min() < 0 else y_true
        y_pred_exp = np.expm1(y_pred) if isinstance(y_pred, np.ndarray) and y_pred.min() < 0 else y_pred
        
        metrics = {
            'rmse': np.sqrt(mean_squared_log_error(y_true_exp, y_pred_exp)),
            'mae': mean_absolute_error(y_true_exp, y_pred_exp),
            'r2': r2_score(y_true_exp, y_pred_exp),
            'mape': np.mean(np.abs((y_true_exp - y_pred_exp) / y_true_exp)) * 100
        }
        
        return metrics
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("Saving models...")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Save selected features
        with open(os.path.join(self.model_dir, 'selected_features.txt'), 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{name}_model.pkl'))
        
        # Save stacked model
        if self.stacked_model:
            joblib.dump(self.stacked_model, os.path.join(self.model_dir, 'stacked_model.pkl'))
        
        # Save feature importance
        joblib.dump(self.feature_importance, os.path.join(self.model_dir, 'feature_importance.pkl'))
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        print("Loading models...")
        
        try:
            # Load scaler
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            
            # Load selected features
            with open(os.path.join(self.model_dir, 'selected_features.txt'), 'r') as f:
                self.selected_features = [line.strip() for line in f.readlines()]
            
            # Load individual models
            for model_file in os.listdir(self.model_dir):
                if model_file.endswith('_model.pkl'):
                    model_name = model_file.replace('_model.pkl', '')
                    self.models[model_name] = joblib.load(os.path.join(self.model_dir, model_file))
            
            # Load stacked model
            stacked_path = os.path.join(self.model_dir, 'stacked_model.pkl')
            if os.path.exists(stacked_path):
                self.stacked_model = joblib.load(stacked_path)
            
            # Load feature importance
            importance_path = os.path.join(self.model_dir, 'feature_importance.pkl')
            if os.path.exists(importance_path):
                self.feature_importance = joblib.load(importance_path)
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    predictor = TaxiDurationPredictor()
    print("Taxi duration predictor initialized!")
