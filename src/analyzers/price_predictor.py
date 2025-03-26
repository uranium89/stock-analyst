import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Dict, Any, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.rf_model = None
        self.xgb_model = None
        
    def prepare_data(self, price_data: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for prediction models"""
        try:
            if price_data is None or len(price_data) < self.sequence_length + 1:
                raise ValueError(f"Not enough data. Need at least {self.sequence_length + 1} days of price data.")
            
            # Ensure we have the required columns
            if 'priceClose' not in price_data.columns:
                raise ValueError("Price data must contain 'priceClose' column")
                
            # Calculate technical indicators
            data = price_data.copy()
            
            # Moving Averages
            for period in [5, 10, 20, 50, 200]:
                data[f'MA{period}'] = data['priceClose'].rolling(window=period, min_periods=1).mean()
            
            # RSI
            data['RSI'] = self._calculate_rsi(data['priceClose'])
            
            # MACD
            data['MACD'] = self._calculate_macd(data['priceClose'])
            
            # Stochastic Oscillator
            data['Stoch_K'], data['Stoch_D'] = self._calculate_stochastic(data)
            
            # ATR
            data['ATR'] = self._calculate_atr(data)
            
            # ADX
            data['ADX'] = self._calculate_adx(data)
            
            # OBV
            data['OBV'] = self._calculate_obv(data)
            
            # CCI
            data['CCI'] = self._calculate_cci(data)
            
            # MFI
            data['MFI'] = self._calculate_mfi(data)
            
            # ROC
            data['ROC'] = self._calculate_roc(data)
            
            # Momentum
            data['MOM'] = self._calculate_momentum(data)
            
            # Williams %R
            data['WILLR'] = self._calculate_williams_r(data)
            
            # Forward fill NaN values for indicators that need historical data
            data = data.fillna(method='ffill')
            
            # Backward fill any remaining NaN values
            data = data.fillna(method='bfill')
            
            # Verify we have enough data after filling NaN values
            if len(data) < self.sequence_length + 1:
                raise ValueError(f"Not enough valid data after processing. Need at least {self.sequence_length + 1} days.")
            
            # Prepare features
            features = [
                'priceClose', 'MA5', 'MA10', 'MA20', 'MA50', 'MA200',
                'RSI', 'MACD', 'Stoch_K', 'Stoch_D', 'ATR', 'ADX',
                'OBV', 'CCI', 'MFI', 'ROC', 'MOM', 'WILLR'
            ]
            X = data[features].values
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_sequences, y = [], []
            for i in range(len(X_scaled) - self.sequence_length):
                X_sequences.append(X_scaled[i:(i + self.sequence_length)])
                y.append(X_scaled[i + self.sequence_length, 0])  # Predict next closing price
                
            if not X_sequences:
                raise ValueError("No valid sequences could be created from the data")
                
            return np.array(X_sequences), np.array(y)
        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['priceLow'].rolling(window=k_period).min()
        high_max = data['priceHigh'].rolling(window=k_period).max()
        k = 100 * ((data['priceClose'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['priceHigh'] - data['priceLow']
        high_close = abs(data['priceHigh'] - data['priceClose'].shift())
        low_close = abs(data['priceLow'] - data['priceClose'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        plus_dm = data['priceHigh'].diff()
        minus_dm = data['priceLow'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(data['priceHigh'] - data['priceLow'])
        tr2 = pd.DataFrame(abs(data['priceHigh'] - data['priceClose'].shift(1)))
        tr3 = pd.DataFrame(abs(data['priceLow'] - data['priceClose'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        return (np.sign(data['priceClose'].diff()) * data['dealVolume']).cumsum()
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (data['priceHigh'] + data['priceLow'] + data['priceClose']) / 3
        tp_ma = tp.rolling(window=period).mean()
        tp_std = tp.rolling(window=period).std()
        return (tp - tp_ma) / (0.015 * tp_std)
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (data['priceHigh'] + data['priceLow'] + data['priceClose']) / 3
        money_flow = typical_price * data['dealVolume']
        
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
        negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        return 100 - (100 / (1 + money_ratio))
    
    def _calculate_roc(self, data: pd.DataFrame, period: int = 12) -> pd.Series:
        """Calculate Rate of Change"""
        return data['priceClose'].pct_change(periods=period) * 100
    
    def _calculate_momentum(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Momentum"""
        return data['priceClose'] - data['priceClose'].shift(period)
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data['priceHigh'].rolling(window=period).max()
        lowest_low = data['priceLow'].rolling(window=period).min()
        return -100 * (highest_high - data['priceClose']) / (highest_high - lowest_low)
    
    def train(self, price_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train the models and return predictions"""
        try:
            # Prepare data with historical data
            X, y = self.prepare_data(price_data, symbol)
            
            # Reshape data for Random Forest
            X_rf = X.reshape(X.shape[0], -1)
            
            # Train Random Forest
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.rf_model.fit(X_rf, y)
            
            # Train XGBoost
            self.xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            self.xgb_model.fit(X_rf, y)
            
            # Make predictions
            last_sequence = X[-1:]
            last_sequence_rf = last_sequence.reshape(1, -1)
            
            rf_pred = self.rf_model.predict(last_sequence_rf)
            xgb_pred = self.xgb_model.predict(last_sequence_rf)
            
            # Ensemble prediction (average of both models)
            predicted_scaled = (rf_pred + xgb_pred) / 2
            
            # Create a zero array with the same number of features
            dummy_array = np.zeros((1, X.shape[2]))
            dummy_array[0, 0] = predicted_scaled[0]
            
            # Inverse transform with the correct shape
            predicted_price = self.scaler.inverse_transform(dummy_array)[0][0]
            
            # Calculate confidence interval based on model agreement
            std_dev = np.std(price_data['priceClose'].pct_change().dropna())
            confidence_interval = {
                'lower': predicted_price * (1 - 2 * std_dev),
                'upper': predicted_price * (1 + 2 * std_dev)
            }
            
            return {
                'predicted_price': predicted_price,
                'confidence_interval': confidence_interval,
                'model_metrics': {
                    'rf_score': self.rf_model.score(X_rf, y),
                    'xgb_score': self.xgb_model.score(X_rf, y)
                }
            }
        except Exception as e:
            raise ValueError(f"Error in price prediction: {str(e)}")
    
    def predict_next_days(self, price_data: pd.DataFrame, symbol: str, days: int = 5) -> List[Dict[str, Any]]:
        """Predict prices for the next n days"""
        try:
            predictions = []
            current_sequence = self.prepare_data(price_data, symbol)[0][-1:]
            current_sequence_rf = current_sequence.reshape(1, -1)
            
            for _ in range(days):
                # Get predictions from both models
                rf_pred = self.rf_model.predict(current_sequence_rf)
                xgb_pred = self.xgb_model.predict(current_sequence_rf)
                
                # Ensemble prediction
                predicted_scaled = (rf_pred + xgb_pred) / 2
                
                # Create a zero array with the same number of features
                dummy_array = np.zeros((1, current_sequence.shape[2]))
                dummy_array[0, 0] = predicted_scaled[0]
                
                # Inverse transform with the correct shape
                predicted_price = self.scaler.inverse_transform(dummy_array)[0][0]
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1, 0] = predicted_scaled[0]
                current_sequence_rf = current_sequence.reshape(1, -1)
                
                # Calculate confidence interval
                std_dev = np.std(price_data['priceClose'].pct_change().dropna())
                confidence_interval = {
                    'lower': predicted_price * (1 - 2 * std_dev),
                    'upper': predicted_price * (1 + 2 * std_dev)
                }
                
                predictions.append({
                    'predicted_price': predicted_price,
                    'confidence_interval': confidence_interval
                })
            
            return predictions
        except Exception as e:
            raise ValueError(f"Error in multi-day prediction: {str(e)}") 