import pandas as pd
import numpy as np
from typing import Dict, Any
from src.config.settings import TECHNICAL_PARAMS
from src.utils.api_client import FireantAPI

class PriceTrendAnalyzer:
    def __init__(self, symbol: str, api_client: FireantAPI = None):
        self.symbol = symbol
        self.api = api_client if api_client else FireantAPI()
        self.price_data = self._fetch_price_data()
        
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
        
        # Calculate Stochastic Oscillator
        low_min = df['priceLow'].rolling(window=TECHNICAL_PARAMS['STOCH_K']).min()
        high_max = df['priceHigh'].rolling(window=TECHNICAL_PARAMS['STOCH_K']).max()
        df['Stoch_K'] = 100 * ((df['priceClose'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=TECHNICAL_PARAMS['STOCH_D']).mean()
        
        # Calculate Average True Range (ATR)
        high_low = df['priceHigh'] - df['priceLow']
        high_close = abs(df['priceHigh'] - df['priceClose'].shift())
        low_close = abs(df['priceLow'] - df['priceClose'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=TECHNICAL_PARAMS['ATR_PERIOD']).mean()
        
        # Calculate Average Directional Index (ADX)
        plus_dm = df['priceHigh'].diff()
        minus_dm = df['priceLow'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['priceHigh'] - df['priceLow'])
        tr2 = pd.DataFrame(abs(df['priceHigh'] - df['priceClose'].shift(1)))
        tr3 = pd.DataFrame(abs(df['priceLow'] - df['priceClose'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(TECHNICAL_PARAMS['ADX_PERIOD']).mean()
        
        plus_di = 100 * (plus_dm.rolling(TECHNICAL_PARAMS['ADX_PERIOD']).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(TECHNICAL_PARAMS['ADX_PERIOD']).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(TECHNICAL_PARAMS['ADX_PERIOD']).mean()
        
        # Calculate On Balance Volume (OBV)
        df['OBV'] = (np.sign(df['priceClose'].diff()) * df['dealVolume']).cumsum()
        
        # Calculate Commodity Channel Index (CCI)
        tp = (df['priceHigh'] + df['priceLow'] + df['priceClose']) / 3
        tp_ma = tp.rolling(window=TECHNICAL_PARAMS['CCI_PERIOD']).mean()
        tp_std = tp.rolling(window=TECHNICAL_PARAMS['CCI_PERIOD']).std()
        df['CCI'] = (tp - tp_ma) / (0.015 * tp_std)
        
        # Calculate Money Flow Index (MFI)
        typical_price = (df['priceHigh'] + df['priceLow'] + df['priceClose']) / 3
        money_flow = typical_price * df['dealVolume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
        negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]
        
        positive_mf = positive_flow.rolling(window=TECHNICAL_PARAMS['MFI_PERIOD']).sum()
        negative_mf = negative_flow.rolling(window=TECHNICAL_PARAMS['MFI_PERIOD']).sum()
        
        money_ratio = positive_mf / negative_mf
        df['MFI'] = 100 - (100 / (1 + money_ratio))
        
        # Calculate Rate of Change (ROC)
        df['ROC'] = df['priceClose'].pct_change(periods=TECHNICAL_PARAMS['ROC_PERIOD']) * 100
        
        # Calculate Momentum
        df['MOM'] = df['priceClose'] - df['priceClose'].shift(TECHNICAL_PARAMS['MOM_PERIOD'])
        
        # Calculate Williams %R
        highest_high = df['priceHigh'].rolling(window=TECHNICAL_PARAMS['WILLR_PERIOD']).max()
        lowest_low = df['priceLow'].rolling(window=TECHNICAL_PARAMS['WILLR_PERIOD']).min()
        df['WILLR'] = -100 * (highest_high - df['priceClose']) / (highest_high - lowest_low)
        
        return df
    
    def analyze_trend(self) -> Dict[str, Any]:
        """Analyze price trends and technical signals"""
        df = self.calculate_technical_indicators()
        latest = df.iloc[-1]
        
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
        
        signals['final_score'] = score
        return signals 