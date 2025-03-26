"""
API client for Fireant API
"""

import requests
from typing import Dict, Any, List
from src.config.settings import FIREANT_TOKEN, FIREANT_BASE_URL
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FireantAPI:
    def __init__(self):
        self.headers = {
            'Authorization': f"Bearer {FIREANT_TOKEN}"
        }
        logger.info("FireantAPI initialized")
    
    def _make_request(self, url: str) -> Dict[str, Any]:
        """Make a request to the Fireant API"""
        logger.info(f"Making API request to: {url}")
        try:
            response = requests.get(url, headers=self.headers)
            logger.info(f"API response status code: {response.status_code}")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.text}")
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            raise
    
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
        """Get historical price data for a symbol"""
        try:
            # Format dates to ensure they match API requirements (dd/MM/yyyy)
            url = f"{FIREANT_BASE_URL}/{symbol}/historical-quotes"
            params = {
                'startDate': start_date,
                'endDate': end_date,
                'offset': str(offset),
                'limit': str(limit)
            }
            
            # Build URL with parameters
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{url}?{query_string}"
            
            logger.info(f"Fetching historical quotes for {symbol} from {start_date} to {end_date} with offset={offset} and limit={limit}")
            logger.info(f"Full URL: {full_url}")
            
            data = self._make_request(full_url)
            logger.info(f"Received {len(data) if data else 0} historical quotes")
            return data
        except Exception as e:
            logger.error(f"Error in get_historical_quotes: {str(e)}")
            raise 