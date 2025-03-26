"""
Configuration settings for AI Trading Analysis
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
FIREANT_TOKEN = os.getenv('FIREANT_TOKEN')

# Model Settings
GEMINI_MODEL = 'gemini-2.0-flash-lite'

# Chart Colors
CHART_COLORS = {
    'primary': '#1f77b4',    # Blue
    'secondary': '#2ca02c',  # Green
    'tertiary': '#ff7f0e',   # Orange
    'danger': '#d62728',     # Red
    'success': '#99FF99',    # Light Green
    'warning': '#FFFF99',    # Light Yellow
    'error': '#FF9999'       # Light Red
}

# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    'MA_PERIODS': [20, 50, 200],
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BB_PERIOD': 20,
    'BB_STD': 2
}

# Scoring Weights
BANK_WEIGHTS = {
    'roe': 0.20,  # Return on Equity
    'roa': 0.15,  # Return on Assets
    'cir': 0.15,  # Cost to Income Ratio
    'clr': 0.10,  # Cost to Loan Ratio
    'ldr': 0.10,  # Loan to Deposit Ratio
    'lar': 0.10,  # Loan to Asset Ratio
    'cost_to_asset': 0.10,  # Cost to Asset Ratio
    'nim': 0.05,  # Net Interest Margin
    'pe': 0.03,   # Price to Earnings Ratio
    'fcf': 0.02   # Free Cash Flow Ratio
}

GENERAL_WEIGHTS = {
    'roe': 0.35,  # Return on Equity
    'fcf': 0.35,  # Free Cash Flow
    'cash': 0.10, # Cash Ratio
    'debt_equity': 0.10,  # Debt to Equity
    'pe': 0.05,   # Price to Earnings
    'equity': 0.05  # Equity Ratio
}

# Analysis Thresholds
ANALYSIS_THRESHOLDS = {
    'bank': {
        'roe': 0.15,  # 15% ROE target
        'roa': 0.01,  # 1% ROA target
        'cir': 0.5,   # 50% CIR target (lower is better)
        'clr': 0.03,  # 3% CLR target (lower is better)
        'ldr': 0.8,   # 80% LDR target
        'lar': 0.6,   # 60% LAR target
        'cost_to_asset': 0.02,  # 2% Cost to Asset target (lower is better)
        'nim': 0.03,  # 3% NIM target
        'pe': 1,      # PE ratio (lower is better)
        'fcf': 0.05   # 5% FCF target
    },
    'general': {
        'roe': 0.20,  # 20% ROE target
        'fcf': 0.10,  # 10% FCF target
        'cash': 0.15, # 15% Cash ratio target
        'debt_equity': 2.0,  # 200% Debt to Equity target (lower is better)
        'pe': 1,      # PE ratio (lower is better)
        'equity': 0.5  # 50% Equity ratio target
    }
}

# API URLs
FIREANT_BASE_URL = "https://api.fireant.vn/symbols" 