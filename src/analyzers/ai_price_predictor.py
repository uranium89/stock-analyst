import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config.settings import TECHNICAL_PARAMS
from src.utils.api_client import FireantAPI

class AIPricePredictor:
    def __init__(self, symbol: str, api_client=None):
        self.symbol = symbol
        self.api = api_client if api_client else FireantAPI()
        self.price_data = self._fetch_price_data()
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _fetch_price_data(self) -> pd.DataFrame:
        """Fetch historical price data"""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=1)
        data = self.api.get_historical_quotes(
            self.symbol,
            start_date.strftime('%m/%d/%Y'),
            end_date.strftime('%m/%d/%Y')
        )
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['date'])
        df = df.sort_values('Date')
        return df
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # Calculate moving averages
        for period in TECHNICAL_PARAMS['MA_PERIODS']:
            df[f'MA{period}'] = df['priceClose'].rolling(window=period).mean()
        
        # Calculate RSI
        delta = df['priceClose'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=TECHNICAL_PARAMS['RSI_PERIOD']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=TECHNICAL_PARAMS['RSI_PERIOD']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['priceClose'].ewm(span=TECHNICAL_PARAMS['MACD_FAST'], adjust=False).mean()
        exp2 = df['priceClose'].ewm(span=TECHNICAL_PARAMS['MACD_SLOW'], adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=TECHNICAL_PARAMS['MACD_SIGNAL'], adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['BB_middle'] = df['priceClose'].rolling(window=TECHNICAL_PARAMS['BB_PERIOD']).mean()
        df['BB_upper'] = df['BB_middle'] + TECHNICAL_PARAMS['BB_STD'] * df['priceClose'].rolling(window=TECHNICAL_PARAMS['BB_PERIOD']).std()
        df['BB_lower'] = df['BB_middle'] - TECHNICAL_PARAMS['BB_STD'] * df['priceClose'].rolling(window=TECHNICAL_PARAMS['BB_PERIOD']).std()
        
        return df
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Calculate future price direction (target)
        df['future_direction'] = (df['priceClose'].shift(-1) > df['priceClose']).astype(int)
        
        # Create feature set with only available columns
        features = []
        
        # Price-based features (only using available columns)
        features.extend([
            'priceClose', 'priceOpen', 'priceHigh', 'priceLow',
            'dealVolume'  # Removed dealValue as it's not available
        ])
        
        # Technical indicators
        for period in TECHNICAL_PARAMS['MA_PERIODS']:
            features.append(f'MA{period}')
        
        features.extend([
            'RSI', 'MACD', 'Signal_Line',
            'BB_middle', 'BB_upper', 'BB_lower'
        ])
        
        # Try to add FCF features if available
        try:
            financial_data = self.api.get_financial_statements(self.symbol)
            if financial_data is not None and not financial_data.empty:
                # Print available columns for debugging
                print(f"Available financial columns: {financial_data.columns.tolist()}")
                
                # Try different possible column names for FCF
                fcf_columns = [
                    'NetProfitFromOperatingActivity',
                    'OperatingCashFlow',
                    'CashFlowFromOperatingActivities',
                    'OperatingActivitiesCashFlow',
                    'FCF',
                    'FreeCashFlow',
                    'CashFlowFromOperatingActivities',  # Common name in financial statements
                    'OperatingActivities',  # Another common name
                    'CashFlowFromOperations'  # Another common name
                ]
                
                # Find the first matching column
                fcf_column = None
                for col in fcf_columns:
                    if col in financial_data.columns:
                        fcf_column = col
                        print(f"Found FCF column: {col}")
                        break
                
                if fcf_column:
                    df['FCF'] = financial_data[fcf_column].values
                    # Calculate FCF metrics
                    df['FCF_MA20'] = df['FCF'].rolling(window=20, min_periods=1).mean()
                    df['FCF_MA50'] = df['FCF'].rolling(window=50, min_periods=1).mean()
                    df['FCF_Change'] = df['FCF'].pct_change()
                    df['FCF_Trend'] = (df['FCF'] > df['FCF_MA20']).astype(int)
                    
                    # Add FCF features to feature list
                    features.extend([
                        'FCF', 'FCF_MA20', 'FCF_MA50',
                        'FCF_Change', 'FCF_Trend'
                    ])
                else:
                    print("Warning: No FCF column found in financial data")
        except Exception as e:
            print(f"Warning: Could not add FCF features: {str(e)}")
        
        # Create lagged features
        for feature in features:
            for lag in [1, 2, 3, 5]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Prepare X and y
        X = df[[col for col in df.columns if 'lag_' in col or col in features]]
        y = df['future_direction']
        
        return X.values, y.values
    
    def train(self) -> Dict[str, float]:
        """Train the XGBoost model"""
        # Get data with technical indicators
        df = self.calculate_technical_indicators(self.price_data)
        
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with updated parameters
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Calculate metrics
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score
        }
    
    def predict(self) -> Dict[str, Any]:
        """Make price direction prediction"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
        
        # Get latest data
        df = self.calculate_technical_indicators(self.price_data)
        
        # Prepare features for prediction
        X, _ = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict_proba(X_scaled[-1:])
        direction_prob = prediction[0][1]  # Probability of price going up
        
        # Calculate confidence score
        confidence = self.calculate_confidence(y_true=X_scaled[-1], y_pred=prediction)
        
        # Determine trend and strength based on probability and confidence
        if direction_prob > 0.6 and confidence > 0.6:
            trend = 'Tăng'
            strength = 'Mạnh'
        elif direction_prob > 0.55 and confidence > 0.5:
            trend = 'Tăng'
            strength = 'Vừa phải'
        elif direction_prob < 0.4 and confidence > 0.6:
            trend = 'Giảm'
            strength = 'Mạnh'
        elif direction_prob < 0.45 and confidence > 0.5:
            trend = 'Giảm'
            strength = 'Vừa phải'
        else:
            trend = 'Trung tính'
            strength = 'Yếu'
        
        return {
            'predicted_return': direction_prob - 0.5,  # Convert to return-like value
            'confidence': confidence,
            'trend': trend,
            'strength': strength,
            'probability': direction_prob
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
        
        # Get feature names
        df = self.calculate_technical_indicators(self.price_data)
        X, _ = self.prepare_features(df)
        
        # Get feature names before converting to numpy array
        feature_names = []
        
        # Price-based features
        base_features = [
            'priceClose', 'priceOpen', 'priceHigh', 'priceLow',
            'dealVolume'
        ]
        
        # Technical indicators
        for period in TECHNICAL_PARAMS['MA_PERIODS']:
            base_features.append(f'MA{period}')
        
        base_features.extend([
            'RSI', 'MACD', 'Signal_Line',
            'BB_middle', 'BB_upper', 'BB_lower'
        ])
        
        # Add FCF features
        base_features.extend([
            'FCF', 'FCF_MA20', 'FCF_MA50',
            'FCF_Change', 'FCF_Trend'
        ])
        
        # Add base features
        feature_names.extend(base_features)
        
        # Add lagged features
        for feature in base_features:
            for lag in [1, 2, 3, 5]:
                feature_names.append(f'{feature}_lag_{lag}')
        
        # Get importance scores
        importance = self.model.feature_importances_
        
        # Create dictionary of feature importance
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return feature_importance
    
    def calculate_confidence(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate confidence level based on multiple factors"""
        # Get the latest data for scoring
        df = self.calculate_technical_indicators(self.price_data)
        X, y = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # 1. Model Performance Score (40% weight)
        # Calculate accuracy on test set
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        model_score = self.model.score(X_test, y_test)
        
        # 2. Recent Accuracy (30% weight)
        # Calculate accuracy on recent predictions
        recent_true = y[-5:]  # Last 5 actual values
        recent_pred = self.model.predict(X_scaled[-5:])  # Last 5 predictions
        recent_accuracy = np.mean(recent_true == recent_pred)
        
        # 3. Prediction Stability (20% weight)
        # Calculate how stable predictions are across recent data points
        recent_predictions = self.model.predict_proba(X_scaled[-5:])  # Last 5 predictions
        prediction_std = np.std(recent_predictions[:, 1])  # Use probability of up direction
        stability_score = 1 / (1 + prediction_std)  # Higher stability = higher score
        
        # 4. Feature Importance Balance (10% weight)
        # Check if predictions rely on multiple features rather than just one
        importance_scores = self.model.feature_importances_
        top_feature_importance = np.max(importance_scores)
        feature_balance = 1 - top_feature_importance  # Higher balance = higher score
        
        # Combine all factors with weights
        confidence = (
            0.4 * model_score +  # Model performance (increased weight)
            0.3 * recent_accuracy +  # Recent accuracy (increased weight)
            0.2 * stability_score +  # Prediction stability
            0.1 * feature_balance  # Feature importance balance (reduced weight)
        )
        
        # Normalize confidence to be between 0 and 1
        confidence = max(0, min(1, confidence))
        
        # Add a minimum threshold to avoid extremely low confidence
        confidence = max(0.01, confidence)
        
        # Scale down confidence if recent accuracy is low
        if recent_accuracy < 0.5:  # If recent predictions are worse than random
            confidence *= 0.7  # Reduce confidence by 30%
        
        return confidence 