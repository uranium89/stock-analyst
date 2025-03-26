import pandas as pd
from typing import Dict, Any, List
from src.config.settings import BANK_WEIGHTS, GENERAL_WEIGHTS, ANALYSIS_THRESHOLDS
from src.utils.api_client import FireantAPI

class CompanyScorer:
    def __init__(self, symbol: str, api_client: FireantAPI = None):
        self.symbol = symbol
        self.api = api_client if api_client else FireantAPI()
        self.raw_yearly_data = self._fetch_data()
        self.fundamental_data = self._fetch_fundamental_data()
        self.profile_data = self._fetch_profile_data()
        self.company_type = self._determine_company_type()
        self.weights = BANK_WEIGHTS if self.company_type == 'bank' else GENERAL_WEIGHTS
        self.data = self._process_data()
        self.payback_analysis = self._calculate_payback_time()
    
    def _fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch yearly financial data"""
        data = self.api.get_financial_data(self.symbol)
        for item in data:
            if 'financialValues' in item:
                financial_values = item['financialValues']
                # Calculate dividend yield if not present
                if 'DividendYield' not in financial_values:
                    price = financial_values.get('PriceAtPeriodEnd', 0)
                    cash_dividend = financial_values.get('CashDividend', 0)
                    if price > 0:
                        financial_values['DividendYield'] = (cash_dividend / price) * 100
                    else:
                        financial_values['DividendYield'] = 0
                
                # Calculate cash dividend if not present
                if 'CashDividend' not in financial_values:
                    net_profit = financial_values.get('NetProfitFromOperatingActivity', 0)
                    if net_profit > 0:
                        financial_values['CashDividend'] = net_profit * 0.3
                    else:
                        financial_values['CashDividend'] = 0
                
                # Set stock dividend to 0 if not present
                if 'StockDividend' not in financial_values:
                    financial_values['StockDividend'] = 0
        return data
    
    def _fetch_fundamental_data(self) -> Dict[str, Any]:
        """Fetch fundamental data"""
        return self.api.get_fundamental_data(self.symbol)
    
    def _fetch_profile_data(self) -> Dict[str, Any]:
        """Fetch profile data"""
        return self.api.get_profile_data(self.symbol)
    
    def _process_data(self) -> Dict[str, Any]:
        """Process and organize financial data"""
        if not self.raw_yearly_data:
            raise Exception("No data available")
        
        yearly_df = pd.DataFrame([item['financialValues'] for item in self.raw_yearly_data])
        yearly_df = yearly_df.sort_values('Year', ascending=True)
        
        # Calculate stock price from market cap and shares outstanding
        market_cap = self.fundamental_data.get('marketCap', 0)
        shares_outstanding = self.fundamental_data.get('sharesOutstanding', 1)
        calculated_price = market_cap / shares_outstanding if shares_outstanding > 0 else 0
        
        # Process fundamental and profile data
        fundamental_info = {
            'company_info': {
                'Tên công ty': self.profile_data.get('companyName', ''),
                'Tên tiếng Anh': self.profile_data.get('companyEnglishName', ''),
                'Mã chứng khoán': self.profile_data.get('symbol', ''),
                'Ngành': self.fundamental_data.get('icbIndustry', ''),
                'Ngành con': self.fundamental_data.get('icbSector', ''),
                'Sàn': self.fundamental_data.get('exchange', ''),
                'Vốn hóa': market_cap,
                'Số lượng CP lưu hành': shares_outstanding,
                'Room nước ngoài': (
                    self.fundamental_data.get('foreignPercent', 0) or 
                    self.fundamental_data.get('foreignOwnership', 0) or 
                    self.fundamental_data.get('foreign_ownership', 0) or 
                    self.fundamental_data.get('foreignRoom', 0)
                ),
                'Beta': self.fundamental_data.get('beta', 0),
                'Giá hiện tại': calculated_price,
                'P/E hiện tại': self.fundamental_data.get('pe', 0)
            },
            'company_profile': {
                'Địa chỉ': self.profile_data.get('address', ''),
                'Điện thoại': self.profile_data.get('phone', ''),
                'Website': self.profile_data.get('website', ''),
                'Người đại diện': self.profile_data.get('representative', ''),
                'Năm thành lập': self.profile_data.get('establishedYear', ''),
                'Ngày niêm yết': self.profile_data.get('listingDate', ''),
                'Vốn điều lệ': self.profile_data.get('charterCapital', 0),
                'Lĩnh vực kinh doanh': self.profile_data.get('businessAreas', []),
                'Giới thiệu': self.profile_data.get('overview', '')
            },
            'financial_metrics': {
                'EPS': self.fundamental_data.get('eps', 0),
                'P/E': self.fundamental_data.get('pe', 0),
                'P/B': self.fundamental_data.get('pb', 0),
                'ROE': self.fundamental_data.get('roe', 0),
                'ROA': self.fundamental_data.get('roa', 0),
                'ROIC': self.fundamental_data.get('roic', 0)
            }
        }
        
        return {
            'latest': yearly_df.iloc[-1].to_dict(),
            'yearly': yearly_df.to_dict('records'),
            'yearly_trends': yearly_df.agg({
                'ROE': ['std', 'mean'],
                'CashflowFromOperatingActivity': ['std', 'mean'],
                'TotalRevenue': ['std', 'mean'],
                'NetProfitFromOperatingActivity': ['std', 'mean']
            }).reset_index().to_dict('records'),
            'time_range': {
                'yearly': {
                    'start': yearly_df['Year'].min(),
                    'end': yearly_df['Year'].max(),
                    'periods': len(yearly_df)
                }
            },
            'fundamental': fundamental_info
        }
    
    def _determine_company_type(self) -> str:
        """Determine if the company is a bank or general company"""
        if not self.raw_yearly_data:
            raise Exception("No data available")
        
        latest = self.raw_yearly_data[0]['financialValues']
        total_debt = latest.get('TotalDebt', 0)
        equity = latest.get('StockHolderEquity', 1)
        interest_income = latest.get('InterestIncome', 0)
        total_revenue = latest.get('TotalRevenue', 1)
        operating_expenses = latest.get('OperatingExpenses', 0)
        
        return 'bank' if (
            total_debt / equity > 5 or
            interest_income / total_revenue > 0.3 or
            operating_expenses / total_revenue > 0.4
        ) else 'general'
    
    def get_metrics(self) -> Dict[str, float]:
        """Get financial metrics for scoring"""
        latest = self.data['latest']
        if self.company_type == 'bank':
            return {
                'roe': latest.get('ROE', 0),
                'roa': latest.get('NetProfitFromOperatingActivity', 0) / latest.get('TotalAsset', 1),
                'cir': latest.get('OperatingExpenses', 0) / latest.get('TotalRevenue', 1),
                'clr': latest.get('OperatingExpenses', 0) / latest.get('TotalLoans', 1),
                'ldr': latest.get('TotalLoans', 0) / latest.get('TotalDeposits', 1),
                'lar': latest.get('TotalLoans', 0) / latest.get('TotalAsset', 1),
                'cost_to_asset': latest.get('OperatingExpenses', 0) / latest.get('TotalAsset', 1),
                'nim': (latest.get('InterestIncome', 0) - latest.get('InterestExpense', 0)) / latest.get('TotalAsset', 1),
                'pe': 1 / (latest.get('PE', 1) + 1),
                'fcf': latest.get('CashflowFromOperatingActivity', 0) / latest.get('TotalAsset', 1)
            }
        else:
            return {
                'roe': latest.get('ROE', 0),
                'fcf': latest.get('CashflowFromOperatingActivity', 0) / latest.get('TotalAsset', 1),
                'cash': (latest.get('Cash', 0) + latest.get('CashEquivalent', 0)) / latest.get('TotalAsset', 1),
                'debt_equity': latest.get('TotalDebt', 0) / latest.get('StockHolderEquity', 1),
                'pe': 1 / (latest.get('PE', 1) + 1),
                'equity': latest.get('StockHolderEquity', 0) / latest.get('TotalAsset', 1)
            }
    
    def normalize_score(self, value: float, metric: str) -> float:
        """Normalize a metric score based on company type and thresholds"""
        thresholds = ANALYSIS_THRESHOLDS[self.company_type]
        
        if metric in ['cir', 'clr', 'cost_to_asset', 'pe']:
            # For metrics where lower is better
            return 1 - min(max(value / thresholds[metric], 0), 1)
        else:
            # For metrics where higher is better
            return min(max(value / thresholds[metric], 0), 1)
    
    def calculate_score(self) -> Dict[str, Any]:
        """Calculate the final score and analysis"""
        metrics = self.get_metrics()
        weighted_scores = {
            metric: self.normalize_score(value, metric) * self.weights[metric]
            for metric, value in metrics.items()
            if metric in self.weights
        }
        
        return {
            'final_score': sum(weighted_scores.values()) * 100,
            'metrics': metrics,
            'weighted_scores': weighted_scores,
            'business_cycle': self.get_business_cycle_analysis()
        }
    
    def get_business_cycle_analysis(self) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze business cycle trends"""
        yearly_data = self.data['yearly']
        analysis = {'yearly_revenue_growth': [], 'yearly_profit_growth': []}
        
        for i in range(len(yearly_data) - 1):
            current, previous = yearly_data[i], yearly_data[i + 1]
            revenue_growth = (current['TotalRevenue'] - previous['TotalRevenue']) / previous['TotalRevenue']
            profit_growth = (current['NetProfitFromOperatingActivity'] - previous['NetProfitFromOperatingActivity']) / previous['NetProfitFromOperatingActivity']
            
            for key, growth in [('yearly_revenue_growth', revenue_growth), ('yearly_profit_growth', profit_growth)]:
                analysis[key].append({
                    'year': current['Year'],
                    'growth': growth,
                    'revenue': current['TotalRevenue'],
                    'profit': current['NetProfitFromOperatingActivity']
                })
        
        return analysis
    
    def _calculate_payback_time(self):
        """Calculate payback time based on cash dividend history and FCF"""
        yearly_data = self.data['yearly']
        current_price = self.data['fundamental']['company_info']['Giá hiện tại']
        
        # Calculate average metrics from last 5 years or available history
        metrics = {
            'dividend_history': [],
            'fcf_history': [],
            'dividend_growth': [],
            'fcf_growth': [],
            'dividend_coverage': []
        }
        
        # Sort data by year in descending order
        sorted_data = sorted(yearly_data, key=lambda x: x['Year'], reverse=True)
        analysis_period = min(5, len(sorted_data))
        
        for i in range(analysis_period):
            year_data = sorted_data[i]
            
            # Get only cash dividend data
            cash_dividend = year_data.get('CashDividend', 0)
            
            # Get FCF data
            fcf = year_data.get('CashflowFromOperatingActivity', 0)
            
            metrics['dividend_history'].append(cash_dividend)
            metrics['fcf_history'].append(fcf)
            
            # Calculate dividend coverage ratio using only cash dividends
            if cash_dividend > 0:
                coverage_ratio = fcf / cash_dividend if cash_dividend != 0 else float('inf')
                metrics['dividend_coverage'].append(coverage_ratio)
        
        # Calculate average metrics
        avg_dividend = sum(metrics['dividend_history']) / len(metrics['dividend_history']) if metrics['dividend_history'] else 0
        avg_fcf = sum(metrics['fcf_history']) / len(metrics['fcf_history']) if metrics['fcf_history'] else 0
        avg_coverage = sum(metrics['dividend_coverage']) / len(metrics['dividend_coverage']) if metrics['dividend_coverage'] else 0
        
        # Calculate dividend growth rate using only cash dividends
        if len(metrics['dividend_history']) > 1:
            dividend_growth_rates = []
            for i in range(len(metrics['dividend_history']) - 1):
                if metrics['dividend_history'][i+1] != 0:
                    growth = (metrics['dividend_history'][i] - metrics['dividend_history'][i+1]) / metrics['dividend_history'][i+1]
                    dividend_growth_rates.append(growth)
            avg_dividend_growth = sum(dividend_growth_rates) / len(dividend_growth_rates) if dividend_growth_rates else 0
        else:
            avg_dividend_growth = 0
        
        # Calculate payback time scenarios using only cash dividends
        conservative_payback = float('inf')
        moderate_payback = float('inf')
        optimistic_payback = float('inf')
        
        if avg_dividend > 0:
            # Conservative: Using historical average cash dividend
            conservative_payback = current_price / avg_dividend
            
            # Moderate: Using historical average with growth
            future_dividend = avg_dividend * (1 + avg_dividend_growth)
            moderate_payback = current_price / future_dividend
            
            # Optimistic: Using FCF potential
            fcf_per_share = avg_fcf / self.data['fundamental']['company_info']['Số lượng CP lưu hành']
            optimistic_payback = current_price / fcf_per_share if fcf_per_share > 0 else float('inf')
        
        return {
            'metrics': {
                'avg_dividend': avg_dividend,
                'avg_fcf': avg_fcf,
                'avg_dividend_growth': avg_dividend_growth,
                'avg_coverage_ratio': avg_coverage
            },
            'payback_years': {
                'conservative': conservative_payback,
                'moderate': moderate_payback,
                'optimistic': optimistic_payback
            },
            'analysis_period': analysis_period,
            'dividend_stability': 'Cao' if avg_coverage >= 2 else 'Trung bình' if avg_coverage >= 1 else 'Thấp',
            'current_price': current_price
        } 