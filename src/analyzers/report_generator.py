import google.generativeai as genai
from typing import Dict, Any
from src.config.settings import GOOGLE_API_KEY, GEMINI_MODEL
import pandas as pd

class ReportGenerator:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def generate_financial_report(self, scorer: Any, result: Dict[str, Any]) -> str:
        """Generate a detailed financial analysis report"""
        company_type_text = "Ngân Hàng" if scorer.company_type == 'bank' else "Công Ty"
        fundamental = scorer.data['fundamental']
        profile = fundamental['company_profile']
        
        prompt = f"""
        Phân tích chi tiết dữ liệu tài chính cho {company_type_text} {scorer.symbol}:
        
        Thông Tin Cơ Bản:
        - Tên công ty: {fundamental['company_info']['Tên công ty']}
        - Tên tiếng Anh: {fundamental['company_info']['Tên tiếng Anh']}
        - Mã chứng khoán: {fundamental['company_info']['Mã chứng khoán']}
        - Sàn: {fundamental['company_info']['Sàn']}
        
        Thông Tin Chi Tiết:
        - Địa chỉ: {profile['Địa chỉ']}
        - Website: {profile['Website']}
        - Người đại diện: {profile['Người đại diện']}
        - Năm thành lập: {profile['Năm thành lập']}
        - Ngày niêm yết: {profile['Ngày niêm yết']}
        - Vốn điều lệ: {profile['Vốn điều lệ']:,.0f}
        
        Lĩnh Vực Kinh Doanh:
        {', '.join(profile['Lĩnh vực kinh doanh'])}
        
        Giới Thiệu Công Ty:
        {profile['Giới thiệu']}
        
        Thông Tin Thị Trường:
        - Vốn hóa: {fundamental['company_info']['Vốn hóa']:,.0f}
        - Số lượng CP lưu hành: {fundamental['company_info']['Số lượng CP lưu hành']:,.0f}
        - Room nước ngoài: {fundamental['company_info']['Room nước ngoài']:.2f}%
        - Beta: {fundamental['company_info']['Beta']:.2f}

        Chỉ Số Tài Chính Cơ Bản:
        - EPS: {fundamental['financial_metrics']['EPS']:.2f}
        - P/E: {fundamental['financial_metrics']['P/E']:.2f}
        - P/B: {fundamental['financial_metrics']['P/B']:.2f}
        - ROE: {fundamental['financial_metrics']['ROE']:.2f}%
        - ROA: {fundamental['financial_metrics']['ROA']:.2f}%
        - ROIC: {fundamental['financial_metrics']['ROIC']:.2f}%
        
        Chỉ Số Tài Chính Chi Tiết:
        - ROE: {result['metrics']['roe']:.4f} (Trọng số: {scorer.weights['roe']*100}%)
        """
        
        if scorer.company_type == 'bank':
            prompt += f"""
        - ROA: {result['metrics']['roa']:.4f} (Trọng số: {scorer.weights['roa']*100}%)
        - Tỷ Lệ Chi Phí/Thu Nhập (CIR): {result['metrics']['cir']:.4f} (Trọng số: {scorer.weights['cir']*100}%)
        - Tỷ Lệ Chi Phí/Cho Vay (CLR): {result['metrics']['clr']:.4f} (Trọng số: {scorer.weights['clr']*100}%)
        - Tỷ Lệ Cho Vay/Huy Động (LDR): {result['metrics']['ldr']:.4f} (Trọng số: {scorer.weights['ldr']*100}%)
        - Tỷ Lệ Cho Vay/Tài Sản (LAR): {result['metrics']['lar']:.4f} (Trọng số: {scorer.weights['lar']*100}%)
        - Tỷ Lệ Chi Phí/Tài Sản: {result['metrics']['cost_to_asset']:.4f} (Trọng số: {scorer.weights['cost_to_asset']*100}%)
        - Biên Lãi Thuần (NIM): {result['metrics']['nim']:.4f} (Trọng số: {scorer.weights['nim']*100}%)
        """
        else:
            prompt += f"""
        - Tỷ Lệ FCF: {result['metrics']['fcf']:.4f} (Trọng số: {scorer.weights['fcf']*100}%)
        - Tỷ Lệ Tiền Mặt: {result['metrics']['cash']:.4f} (Trọng số: {scorer.weights['cash']*100}%)
        - Nợ/Vốn Chủ Sở Hữu: {result['metrics']['debt_equity']:.4f} (Trọng số: {scorer.weights['debt_equity']*100}%)
        - Tỷ Số P/E: {scorer.data['latest'].get('PE', 0):.2f} (Trọng số: {scorer.weights['pe']*100}%)
        - Tỷ Lệ Vốn Chủ Sở Hữu: {result['metrics']['equity']:.4f} (Trọng số: {scorer.weights['equity']*100}%)
        """

        prompt += f"""
        Thông Tin Công Ty:
        - Tổng Tài Sản: {scorer.data['latest'].get('TotalAsset', 0):,.0f}
        - Tổng Doanh Thu: {scorer.data['latest'].get('TotalRevenue', 0):,.0f}
        - Lợi Nhuận Ròng: {scorer.data['latest'].get('NetProfitFromOperatingActivity', 0):,.0f}
        - Vốn Hóa Thị Trường: {scorer.data['latest'].get('MarketCapAtPeriodEnd', 0):,.0f}
        - Giá Cổ Phiếu: {scorer.data['latest'].get('PriceAtPeriodEnd', 0):,.0f}
        - Tỷ Lệ Cổ Tức: {scorer.data['latest'].get('DividendYield', 0):,.2f}%
        - P/E: {scorer.data['latest'].get('PE', 0):,.2f}
        - EPS: {scorer.data['latest'].get('BasicEPS', 0):,.2f}
        - Payback: {scorer.payback_analysis['payback_years']['moderate']:.1f} năm

        Phân Tích Chu Kỳ Kinh Doanh:
        {pd.DataFrame(result['business_cycle']['yearly_revenue_growth']).to_string()}
        {pd.DataFrame(result['business_cycle']['yearly_profit_growth']).to_string()}

        Điểm Số Tổng Thể: {result['final_score']:.2f}/100

        Hãy cung cấp phân tích toàn diện bao gồm:
        1. Phân Tích Chính:
           - Phân tích chi tiết hiệu suất và xu hướng ROE
        """
        
        if scorer.company_type == 'bank':
            prompt += """
           - Phân tích hiệu quả hoạt động qua ROA và NIM
           - Đánh giá chất lượng quản lý chi phí qua CIR và CLR
           - Phân tích cấu trúc cho vay qua LDR và LAR
        """
        else:
            prompt += """
           - Khả năng tạo và hiệu quả của FCF
           - So sánh các chỉ số này với tiêu chuẩn ngành
        """
        
        if scorer.company_type == 'bank':
            prompt += """
        2. Phân Tích Ngân Hàng:
           - Hiệu quả quản lý chi phí (CIR, CLR)
           - Chất lượng tài sản và quản lý rủi ro
           - Khả năng tạo lợi nhuận từ vốn chủ sở hữu
           - Đánh giá cấu trúc cho vay và huy động
           - Phân tích biên lãi thuần và hiệu quả hoạt động
        """
        else:
            prompt += """
        2. Phân Tích Chu Kỳ Kinh Doanh:
           - Mẫu hình tăng trưởng doanh thu
           - Xu hướng tăng trưởng lợi nhuận
           - Mẫu hình theo mùa (nếu có)
           - Vị trí trong chu kỳ kinh doanh
        """
        
        prompt += """
        3. Phân Tích Hỗ Trợ:
           - Vị thế tiền mặt và thanh khoản
           - Cấu trúc nợ và đòn bẩy
           - Chỉ số định giá
         
        4. Đánh Giá Đầu Tư:
           - Điểm mạnh và điểm yếu dựa trên các chỉ số chính
           - Tiềm năng tăng trưởng và tính bền vững
           - Yếu tố rủi ro và chiến lược giảm thiểu
         
        5. Khuyến Nghị:
           - Triển vọng ngắn hạn và dài hạn
           - Các lĩnh vực cần cải thiện
           - Cân nhắc đầu tư
        """
        
        # Add payback analysis to the prompt
        payback = scorer.payback_analysis
        
        prompt += f"""
        Phân Tích Thời Gian Hoàn Vốn:
        - Giá hiện tại: {payback['current_price']:,.0f}
        - Cổ tức trung bình {payback['analysis_period']} năm gần nhất: {payback['metrics']['avg_dividend']:,.0f}
        - Tăng trưởng cổ tức trung bình: {payback['metrics']['avg_dividend_growth']:.2%}
        - Mức độ ổn định cổ tức: {payback['dividend_stability']}
        
        Thời gian hoàn vốn dự kiến:
        - Kịch bản thận trọng: {payback['payback_years']['conservative']:.1f} năm
        - Kịch bản cơ sở: {payback['payback_years']['moderate']:.1f} năm
        - Kịch bản lạc quan: {payback['payback_years']['optimistic']:.1f} năm
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi khi tạo báo cáo: {str(e)}"
    
    def generate_price_analysis_report(self, price_analysis: Dict[str, Any], price_data: pd.DataFrame) -> str:
        """Generate a detailed price trend analysis report"""
        latest = price_data.iloc[-1]
        
        prompt = f"""
        Phân tích chi tiết xu hướng giá:

        1. Tín Hiệu Xu Hướng Giá:
        - Ngắn hạn (MA20): {price_analysis['price_trend']['short_term']}
        - Trung hạn (MA50): {price_analysis['price_trend']['medium_term']}
        - Dài hạn (MA200): {price_analysis['price_trend']['long_term']}

        2. Chỉ Báo Kỹ Thuật:
        - RSI: {price_analysis['technical_signals']['rsi']['value']:.2f} - {price_analysis['technical_signals']['rsi']['signal']}
        - MACD: {price_analysis['technical_signals']['macd']['value']:.2f} - {price_analysis['technical_signals']['macd']['signal']}
        - Bollinger Bands: {price_analysis['technical_signals']['bollinger']['signal']}

        3. Phân Tích Khối Lượng:
        - Xu hướng khối lượng: {price_analysis['volume_analysis']['trend']}
        - Khối lượng hiện tại: {price_analysis['volume_analysis']['value']:,.0f}

        4. Thông Tin Giá:
        - Giá hiện tại: {latest['priceClose']:,.0f}
        - Giá cao nhất (52 tuần): {price_data['priceHigh'].max():,.0f}
        - Giá thấp nhất (52 tuần): {price_data['priceLow'].min():,.0f}
        - Biến động giá 20 ngày: {(price_data['priceClose'].pct_change().rolling(window=20).std() * 100).iloc[-1]:.2f}%

        Điểm số xu hướng: {price_analysis['final_score']:.2f}/100

        Hãy cung cấp phân tích chi tiết bao gồm:
        1. Tổng Quan Xu Hướng:
           - Phân tích xu hướng chính của giá
           - Đánh giá mức độ mạnh/yếu của xu hướng
           - Các mức hỗ trợ và kháng cự quan trọng

        2. Phân Tích Kỹ Thuật:
           - Đánh giá tín hiệu từ RSI và MACD
           - Phân tích vị trí giá trong dải Bollinger
           - Nhận định về động lượng của giá

        3. Phân Tích Khối Lượng:
           - Đánh giá sự xác nhận của khối lượng với xu hướng
           - Phân tích sự thay đổi của khối lượng giao dịch
           - Các điểm bất thường về khối lượng

        4. Dự Báo Xu Hướng:
           - Triển vọng ngắn hạn (1-2 tuần)
           - Triển vọng trung hạn (1-3 tháng)
           - Các yếu tố cần theo dõi

        5. Khuyến Nghị:
           - Đề xuất hành động cụ thể
           - Các mức giá quan trọng cần chú ý
           - Chiến lược giao dịch phù hợp
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi khi tạo báo cáo xu hướng giá: {str(e)}"
    
    def generate_market_report(self, df: pd.DataFrame, ai_prediction: Dict[str, Any], ai_metrics: Dict[str, Any], ai_importance: Dict[str, float]) -> str:
        """Generate a detailed market analysis report using Gemini AI"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate market trend
        ma20 = df['MA20'].iloc[-1]
        ma50 = df['MA50'].iloc[-1]
        ma200 = df['MA200'].iloc[-1]
        current_price = df['priceClose'].iloc[-1]
        
        # Determine market trend
        if current_price > ma20 > ma50 > ma200:
            trend = "Xu hướng tăng mạnh"
        elif current_price > ma20 > ma50:
            trend = "Xu hướng tăng"
        elif current_price < ma20 < ma50 < ma200:
            trend = "Xu hướng giảm mạnh"
        elif current_price < ma20 < ma50:
            trend = "Xu hướng giảm"
        else:
            trend = "Xu hướng đi ngang"
        
        # Calculate volume trend
        volume_ma20 = df['dealVolume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['dealVolume'].iloc[-1]
        volume_trend = "Cao hơn" if current_volume > volume_ma20 else "Thấp hơn"
        
        # Prepare prompt for Gemini
        prompt = f"""
        Phân tích chi tiết thị trường chứng khoán Việt Nam (VNINDEX):
        
        Thông Tin Cơ Bản:
        - Giá đóng cửa: {latest['priceClose']:,.2f}
        - Thay đổi: {((latest['priceClose'] - prev['priceClose']) / prev['priceClose'] * 100):.2f}%
        - Khối lượng: {current_volume:,.0f} ({volume_trend} trung bình 20 phiên)
        
        Xu Hướng Thị Trường:
        - Xu hướng hiện tại: {trend}
        - RSI: {latest['RSI']:.2f} ({'Quá mua' if latest['RSI'] > 70 else 'Quá bán' if latest['RSI'] < 30 else 'Trung tính'})
        - MACD: {'Tín hiệu mua' if latest['MACD'] > latest['Signal_Line'] else 'Tín hiệu bán'}
        
        Chỉ Báo Kỹ Thuật:
        - MA20: {ma20:,.2f}
        - MA50: {ma50:,.2f}
        - MA200: {ma200:,.2f}
        - Bollinger Bands: {'Quá mua' if current_price > latest['BB_upper'] else 'Quá bán' if current_price < latest['BB_lower'] else 'Trung tính'}
        
        Dự Đoán AI:
        - Xu hướng dự đoán: {ai_prediction['trend']}
        - Độ tin cậy: {ai_prediction['confidence']:.2%}
        - Sức mạnh tín hiệu: {ai_prediction['strength']}
        - Độ chính xác mô hình: {ai_metrics['test_score']:.2%}
        
        Các Chỉ Số Quan Trọng Nhất:
        {', '.join([f"{k}: {v:.2f}" for k, v in list(ai_importance.items())[:5]])}
        
        Hãy phân tích chi tiết các yếu tố trên và đưa ra nhận định về xu hướng thị trường trong thời gian tới.
        """
        
        # Generate report using Gemini
        response = self.model.generate_content(prompt)
        return response.text 