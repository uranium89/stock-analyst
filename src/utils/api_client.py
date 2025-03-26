"""
API client for Fireant API
"""

import requests
from typing import Dict, Any, List
from src.config.settings import FIREANT_TOKEN, FIREANT_BASE_URL

class FireantAPI:
    def __init__(self):
        self.headers = {
            'Authorization': f"Bearer {FIREANT_TOKEN}"
        }
    
    def _make_request(self, url: str) -> Dict[str, Any]:
        """Make a request to the Fireant API"""
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def get_financial_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get yearly financial data for a symbol"""
        url = f"{FIREANT_BASE_URL}/{symbol}/financial-data?type=Y&count=20"
        return self._make_request(url)
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol"""
        url = f"{FIREANT_BASE_URL}/{symbol}/fundamental"
        return self._make_request(url)
    
    def get_profile_data(self, symbol: str) -> Dict[str, Any]:
        """Get profile data for a symbol"""
        url = f"{FIREANT_BASE_URL}/{symbol}/profile"
        return self._make_request(url)
    
    def get_historical_quotes(self, symbol: str, start_date: str, end_date: str, offset: int = 0, limit: int = 365) -> List[Dict[str, Any]]:
        """Get historical price data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date in MM/DD/YYYY format
            end_date: End date in MM/DD/YYYY format
            offset: Number of records to skip (default: 0)
            limit: Maximum number of records to return (default: 365)
            
        Returns:
            List of historical price data
        """
        url = f"{FIREANT_BASE_URL}/{symbol}/historical-quotes?startDate={start_date}&endDate={end_date}&offset={offset}&limit={limit}"
        return self._make_request(url) 