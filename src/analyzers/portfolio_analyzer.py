from typing import List, Dict
import pandas as pd
from .company_scorer import CompanyScorer

class PortfolioAnalyzer:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.analyzers = {}
        self.scores = {}
        self.payback_times = {}
        
    def analyze_portfolio(self) -> Dict:
        """Analyze portfolio and suggest allocation based on scores and payback times"""
        # Analyze each company
        for symbol in self.symbols:
            try:
                scorer = CompanyScorer(symbol)
                financial_result = scorer.calculate_score()
                
                # Store analysis results
                self.analyzers[symbol] = scorer
                self.scores[symbol] = financial_result['final_score']
                self.payback_times[symbol] = scorer.payback_analysis['payback_years']['moderate']
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Calculate portfolio weights
        weights = self._calculate_weights()
        
        # Generate portfolio summary
        summary = self._generate_summary(weights)
        
        return {
            'weights': weights,
            'summary': summary,
            'scores': self.scores,
            'payback_times': self.payback_times
        }
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate portfolio weights based on scores and payback times"""
        if not self.scores or not self.payback_times:
            return {}
            
        # Normalize scores (0-100 to 0-1)
        max_score = max(self.scores.values())
        normalized_scores = {symbol: score/max_score for symbol, score in self.scores.items()}
        
        # Normalize payback times (inverse relationship - shorter is better)
        # Handle infinite or very large payback times
        valid_payback_times = {symbol: time for symbol, time in self.payback_times.items() 
                             if time is not None and time != float('inf') and time > 0}
        
        if valid_payback_times:
            max_payback = max(valid_payback_times.values())
            normalized_payback = {symbol: 1 - (time/max_payback) for symbol, time in valid_payback_times.items()}
        else:
            normalized_payback = {symbol: 0.5 for symbol in self.scores}  # Default value if no valid payback times
        
        # Combine scores (70% weight on financial score, 30% on payback time)
        combined_scores = {}
        for symbol in self.scores:
            # Use 0.5 as default payback score for symbols with invalid payback times
            payback_score = normalized_payback.get(symbol, 0.5)
            combined_scores[symbol] = (
                0.7 * normalized_scores[symbol] +
                0.3 * payback_score
            )
        
        # Calculate weights (normalize to sum to 1)
        total_score = sum(combined_scores.values())
        if total_score > 0:
            weights = {symbol: score/total_score for symbol, score in combined_scores.items()}
        else:
            # If all scores are 0, distribute weights equally
            weights = {symbol: 1/len(self.scores) for symbol in self.scores}
        
        return weights
    
    def _generate_summary(self, weights: Dict[str, float]) -> Dict:
        """Generate portfolio summary with key metrics"""
        summary = {
            'total_companies': len(self.symbols),
            'average_score': sum(self.scores.values()) / len(self.scores),
            'average_payback': sum(self.payback_times.values()) / len(self.payback_times),
            'portfolio_metrics': {}
        }
        
        # Calculate portfolio metrics
        for symbol in self.symbols:
            analyzer = self.analyzers[symbol]
            weight = weights[symbol]
            
            # Get company metrics
            company_info = analyzer.data['fundamental']['company_info']
            financial_metrics = analyzer.data['fundamental']['financial_metrics']
            latest = analyzer.data['latest']
            
            # Calculate weighted metrics
            summary['portfolio_metrics'][symbol] = {
                'weight': weight,
                'score': self.scores[symbol],
                'payback_time': self.payback_times[symbol],
                'market_cap': company_info.get('Vốn hóa', 0),
                'pe_ratio': financial_metrics.get('P/E', 0),
                'roe': latest.get('ROE', 0),
                'dividend_yield': latest.get('DividendYield', 0)
            }
        
        return summary 