import pandas as pd
from typing import Dict, Any
from src.config.settings import TECHNICAL_PARAMS
from src.utils.api_client import FireantAPI
from src.analyzers.ai_price_predictor import AIPricePredictor

class PriceTrendAnalyzer:
    def __init__(self, symbol: str, api_client: FireantAPI = None):
        self.symbol = symbol
        self.api = api_client if api_client else FireantAPI()
        self.price_data = self._fetch_price_data()
        self.ai_predictor = AIPricePredictor(symbol, api_client)
        
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
    
    def calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = self.price_data.copy()
        
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
    
    def analyze_trend(self) -> Dict[str, Any]:
        """Analyze price trends and technical signals"""
        df = self.calculate_technical_indicators()
        latest = df.iloc[-1]
        
        # Train AI model and get prediction
        try:
            ai_metrics = self.ai_predictor.train()
            ai_prediction = self.ai_predictor.predict()
            ai_importance = self.ai_predictor.get_feature_importance()
        except Exception as e:
            print(f"AI prediction failed: {str(e)}")
            ai_prediction = None
            ai_metrics = None
            ai_importance = None
        
        # Trend signals
        signals = {
            'price_trend': {
                'short_term': 'Tăng' if latest['priceClose'] > latest['MA20'] else 'Giảm',
                'medium_term': 'Tăng' if latest['priceClose'] > latest['MA50'] else 'Giảm',
                'long_term': 'Tăng' if latest['priceClose'] > latest['MA200'] else 'Giảm'
            },
            'technical_signals': {
                'rsi': {
                    'value': latest['RSI'],
                    'signal': 'Quá mua' if latest['RSI'] > 70 else 'Quá bán' if latest['RSI'] < 30 else 'Trung tính'
                },
                'macd': {
                    'value': latest['MACD'],
                    'signal': 'Mua' if latest['MACD'] > latest['Signal_Line'] else 'Bán'
                },
                'bollinger': {
                    'value': latest['priceClose'],
                    'signal': 'Quá mua' if latest['priceClose'] > latest['BB_upper'] else 'Quá bán' if latest['priceClose'] < latest['BB_lower'] else 'Trung tính'
                }
            },
            'volume_analysis': {
                'value': latest['dealVolume'],
                'trend': 'Tăng' if df['dealVolume'].iloc[-5:].mean() > df['dealVolume'].iloc[-20:-5].mean() else 'Giảm'
            }
        }
        
        # Add AI prediction if available
        if ai_prediction:
            signals['ai_prediction'] = {
                'trend': ai_prediction['trend'],
                'predicted_return': ai_prediction['predicted_return'],
                'confidence': ai_prediction['confidence'],
                'strength': ai_prediction['strength'],
                'model_performance': ai_metrics,
                'top_features': dict(list(ai_importance.items())[:5])
            }
        
        # Calculate trend score (0-100)
        score = 0
        
        # Price trend contribution (40 points)
        if signals['price_trend']['short_term'] == 'Tăng': score += 15
        if signals['price_trend']['medium_term'] == 'Tăng': score += 15
        if signals['price_trend']['long_term'] == 'Tăng': score += 10
        
        # Technical signals contribution (40 points)
        # RSI (15 points)
        if 40 <= signals['technical_signals']['rsi']['value'] <= 60:
            score += 15
        elif 30 <= signals['technical_signals']['rsi']['value'] <= 70:
            score += 10
        
        # MACD (15 points)
        if signals['technical_signals']['macd']['signal'] == 'Mua':
            score += 15
        
        # Bollinger Bands (10 points)
        if signals['technical_signals']['bollinger']['signal'] == 'Trung tính':
            score += 10
        elif signals['technical_signals']['bollinger']['signal'] == 'Quá bán':
            score += 5
        
        # Volume trend contribution (20 points)
        if signals['volume_analysis']['trend'] == 'Tăng':
            score += 20
        
        # Add AI prediction contribution if available
        if ai_prediction:
            if ai_prediction['trend'] == 'Tăng':
                score += 10
            if ai_prediction['strength'] == 'Mạnh':
                score += 10
        
        signals['final_score'] = score
        return signals 