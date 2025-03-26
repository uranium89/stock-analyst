import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.config.settings import CHART_COLORS, TECHNICAL_PARAMS

class ChartCreator:
    @staticmethod
    def create_gauge_chart(score: float, title: str) -> go.Figure:
        """Create a gauge chart for displaying scores"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "darkblue",
                    'tickmode': 'linear',
                    'dtick': 20
                },
                'bar': {'color': "darkblue", 'thickness': 0.5},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': CHART_COLORS['error']},
                    {'range': [33, 66], 'color': CHART_COLORS['warning']},
                    {'range': [66, 100], 'color': CHART_COLORS['success']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            },
            number={'font': {'size': 40, 'color': 'darkblue'}, 'suffix': '/100'}
        ))
        
        return fig.update_layout(
            height=300,
            margin={'t': 40, 'b': 20, 'l': 20, 'r': 20},
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )

    @staticmethod
    def create_trend_chart(data: List[Dict[str, Any]], title: str, y_label: str) -> go.Figure:
        """Create a trend chart for revenue and profit growth"""
        fig = go.Figure()
        
        for name, color, y_key in [
            ('Doanh Thu', CHART_COLORS['primary'], 'revenue'),
            ('Lợi Nhuận', CHART_COLORS['secondary'], 'profit')
        ]:
            fig.add_trace(go.Scatter(
                x=[item['year'] for item in data],
                y=[item[y_key] for item in data],
                name=name,
                line=dict(color=color, width=2),
                mode='lines+markers'
            ))
        
        return fig.update_layout(
            title=title,
            xaxis_title='Năm',
            yaxis_title=y_label,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        )

    @staticmethod
    def create_roe_chart(yearly_data: List[Dict[str, Any]]) -> go.Figure:
        """Create a ROE trend chart"""
        fig = go.Figure()
        
        # Add ROE trend
        fig.add_trace(go.Scatter(
            x=[item['Year'] for item in yearly_data],
            y=[item['ROE'] * 100 for item in yearly_data],
            name='ROE',
            line=dict(color=CHART_COLORS['primary'], width=2),
            mode='lines+markers'
        ))
        
        # Add trend line
        x = [item['Year'] for item in yearly_data]
        y = [item['ROE'] * 100 for item in yearly_data]
        z = np.polyfit(range(len(x)), y, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=x,
            y=p(range(len(x))),
            name='Đường Xu Hướng ROE',
            line=dict(color=CHART_COLORS['primary'], width=1, dash='dash'),
            mode='lines'
        ))
        
        return fig.update_layout(
            title="Xu Hướng ROE",
            xaxis_title='Năm',
            yaxis_title='ROE (%)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        )

    @staticmethod
    def create_equity_chart(yearly_data: List[Dict[str, Any]]) -> go.Figure:
        """Create an equity ratio trend chart"""
        fig = go.Figure()
        
        # Add Equity trend
        fig.add_trace(go.Scatter(
            x=[item['Year'] for item in yearly_data],
            y=[item['StockHolderEquity'] / item['TotalAsset'] * 100 for item in yearly_data],
            name='Tỷ Lệ Vốn Chủ Sở Hữu',
            line=dict(color=CHART_COLORS['secondary'], width=2),
            mode='lines+markers'
        ))
        
        # Add trend line
        x = [item['Year'] for item in yearly_data]
        y = [item['StockHolderEquity'] / item['TotalAsset'] * 100 for item in yearly_data]
        z = np.polyfit(range(len(x)), y, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=x,
            y=p(range(len(x))),
            name='Đường Xu Hướng Vốn Chủ Sở Hữu',
            line=dict(color=CHART_COLORS['secondary'], width=1, dash='dash'),
            mode='lines'
        ))
        
        return fig.update_layout(
            title="Xu Hướng Tỷ Lệ Vốn Chủ Sở Hữu",
            xaxis_title='Năm',
            yaxis_title='Tỷ Lệ Vốn Chủ Sở Hữu (%)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        )

    @staticmethod
    def create_dividend_chart(yearly_data: List[Dict[str, Any]]) -> go.Figure:
        """Create a dividend trend chart"""
        fig = go.Figure()
        
        # Add Dividend Yield trend
        fig.add_trace(go.Scatter(
            x=[item['Year'] for item in yearly_data],
            y=[item['DividendYield'] for item in yearly_data],
            name='Tỷ Suất Cổ Tức',
            line=dict(color=CHART_COLORS['tertiary'], width=2),
            mode='lines+markers'
        ))
        
        # Add Cash Dividend trend
        fig.add_trace(go.Scatter(
            x=[item['Year'] for item in yearly_data],
            y=[item['CashDividend'] for item in yearly_data],
            name='Cổ Tức Tiền Mặt',
            line=dict(color=CHART_COLORS['secondary'], width=2),
            mode='lines+markers'
        ))
        
        # Add Stock Dividend trend
        fig.add_trace(go.Scatter(
            x=[item['Year'] for item in yearly_data],
            y=[item['StockDividend'] for item in yearly_data],
            name='Cổ Tức Cổ Phiếu',
            line=dict(color=CHART_COLORS['danger'], width=2),
            mode='lines+markers'
        ))
        
        # Add trend lines
        metrics = [
            ('DividendYield', CHART_COLORS['tertiary']),
            ('CashDividend', CHART_COLORS['secondary']),
            ('StockDividend', CHART_COLORS['danger'])
        ]
        
        for metric, color in metrics:
            x = [item['Year'] for item in yearly_data]
            y = [item[metric] for item in yearly_data]
            z = np.polyfit(range(len(x)), y, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=x,
                y=p(range(len(x))),
                name=f'Đường Xu Hướng {metric}',
                line=dict(color=color, width=1, dash='dash'),
                mode='lines'
            ))
        
        return fig.update_layout(
            title="Xu Hướng Cổ Tức",
            xaxis_title='Năm',
            yaxis_title='Giá Trị (%)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        )

    @staticmethod
    def create_price_trend_chart(price_data: pd.DataFrame) -> go.Figure:
        """Create a price trend chart with technical indicators"""
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=price_data['Date'],
            open=price_data['priceOpen'],
            high=price_data['priceHigh'],
            low=price_data['priceLow'],
            close=price_data['priceClose'],
            name='Giá'
        ))
        
        # Add moving averages
        for ma, color in [
            ('MA20', CHART_COLORS['primary']),
            ('MA50', CHART_COLORS['secondary']),
            ('MA200', CHART_COLORS['tertiary'])
        ]:
            fig.add_trace(go.Scatter(
                x=price_data['Date'],
                y=price_data[ma],
                name=ma,
                line=dict(color=color, width=1)
            ))
        
        # Add Bollinger Bands
        for band, color in [('BB_upper', 'rgba(0,0,0,0.3)'), ('BB_lower', 'rgba(0,0,0,0.3)')]:
            fig.add_trace(go.Scatter(
                x=price_data['Date'],
                y=price_data[band],
                name=band,
                line=dict(color=color, width=1, dash='dash')
            ))
        
        return fig.update_layout(
            title="Biểu Đồ Giá và Chỉ Báo Kỹ Thuật",
            xaxis_title="Ngày",
            yaxis_title="Giá",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGrey'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        )

    @staticmethod
    def create_technical_indicators_chart(price_data: pd.DataFrame) -> go.Figure:
        """Create a technical indicators chart"""
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=("RSI", "MACD"),
                          vertical_spacing=0.2)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=price_data['Date'],
            y=price_data['RSI'],
            name='RSI',
            line=dict(color=CHART_COLORS['primary'])
        ), row=1, col=1)
        
        # Add RSI levels
        for level in [30, 70]:
            fig.add_hline(y=level, line_dash="dash", line_color="gray", row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=price_data['Date'],
            y=price_data['MACD'],
            name='MACD',
            line=dict(color=CHART_COLORS['secondary'])
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=price_data['Date'],
            y=price_data['Signal_Line'],
            name='Signal Line',
            line=dict(color=CHART_COLORS['tertiary'])
        ), row=2, col=1)
        
        return fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=50, b=50)
        ) 