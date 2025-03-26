import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
from plotly.subplots import make_subplots
from src.analyzers.company_scorer import CompanyScorer
from src.analyzers.price_analyzer import PriceTrendAnalyzer
from src.analyzers.report_generator import ReportGenerator
from src.analyzers.portfolio_analyzer import PortfolioAnalyzer
from src.analyzers.price_predictor import PricePredictor
from src.charts.chart_creator import ChartCreator
from src.utils.api_client import FireantAPI
from src.utils.symbol_validator import SymbolValidator
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

def main():
    st.set_page_config(
        page_title="AI Trading Analysis",
        page_icon="📈",
        layout="wide"
    )
    
    # Add sidebar menu
    with st.sidebar:
        st.title("Menu")
        menu_option = st.radio(
            "Chọn chức năng",
            ["Phân Tích Cổ Phiếu", "Phân Bổ Danh Mục", "Trợ Lý Đầu Tư"]
        )
    
    if menu_option == "Phân Tích Cổ Phiếu":
        st.title("AI Trading Analysis")
        st.write("Phân tích tài chính và xu hướng giá tự động sử dụng AI")
        
        # Initialize components
        api_client = FireantAPI()
        chart_creator = ChartCreator()
        report_generator = ReportGenerator()
        symbol_validator = SymbolValidator()
        
        # Get user input
        symbol = st.text_input("Nhập mã chứng khoán (VD: VNM, VCB, VHM):", "").upper()
        
        if symbol:
            # Validate symbol using Gemini
            validation_result = symbol_validator.validate_symbol(symbol)
            
            if not validation_result["is_valid"]:
                st.error(f"Mã chứng khoán không hợp lệ: {validation_result['reason']}")
                return
                
            st.success(f"Mã chứng khoán hợp lệ: {symbol} - {validation_result['company_name']} ({validation_result['exchange']})")
            
            # Continue with existing analysis code
            try:
                with st.spinner("Đang phân tích..."):
                    # Initialize analyzers
                    scorer = CompanyScorer(symbol, api_client)
                    price_analyzer = PriceTrendAnalyzer(symbol, api_client)
                    price_predictor = PricePredictor()
                    
                    # Get analysis results
                    financial_result = scorer.calculate_score()
                    price_analysis = price_analyzer.analyze()
                    price_prediction = price_predictor.train(price_analyzer.price_data, symbol)
                    
                    # Generate reports
                    financial_report = report_generator.generate_financial_report(scorer, financial_result)
                    price_report = report_generator.generate_price_analysis_report(price_analysis, price_analyzer.price_data)
                    
                    # Display results
                    # ... rest of the existing display code ...
            
            except Exception as e:
                logger.error(f"Error in main analysis: {str(e)}")
                st.error(f"Lỗi khi phân tích: {str(e)}")
                st.error("Vui lòng kiểm tra lại mã chứng khoán và thử lại.")
    elif menu_option == "Phân Bổ Danh Mục":
        st.title("Phân Bổ Danh Mục Đầu Tư")
        st.write("Phân tích và đề xuất phân bổ danh mục dựa trên điểm số tài chính và thời gian hoàn vốn")
        
        # Get list of symbols from user
        symbols_input = st.text_area(
            "Nhập danh sách mã chứng khoán (phân cách bằng dấu phẩy):",
            "VNM, VCB, VHM, FPT, VIC"
        )
        
        # Process input
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
        
        if st.button("Phân Tích Danh Mục"):
            try:
                with st.spinner("Đang phân tích danh mục..."):
                    # Initialize portfolio analyzer
                    portfolio_analyzer = PortfolioAnalyzer(symbols)
                    portfolio_result = portfolio_analyzer.analyze_portfolio()
                    
                    # Display portfolio summary
                    st.subheader("Tổng Quan Danh Mục")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Số Công Ty",
                            portfolio_result['summary']['total_companies']
                        )
                    
                    with col2:
                        st.metric(
                            "Điểm Số Trung Bình",
                            f"{portfolio_result['summary']['average_score']:.2f}/100"
                        )
                    
                    with col3:
                        st.metric(
                            "Thời Gian Hoàn Vốn TB",
                            f"{portfolio_result['summary']['average_payback']:.1f} năm"
                        )
                    
                    # Display portfolio allocation
                    st.subheader("Đề Xuất Phân Bổ Danh Mục")
                    
                    # Create DataFrame for portfolio metrics
                    portfolio_data = []
                    for symbol, metrics in portfolio_result['summary']['portfolio_metrics'].items():
                        portfolio_data.append({
                            'Mã CK': symbol,
                            'Tỷ Trọng (%)': f"{metrics['weight'] * 100:.1f}%",
                            'Điểm Số': f"{metrics['score']:.2f}/100",
                            'Thời Gian HV': f"{metrics['payback_time']:.1f} năm",
                            'P/E': f"{metrics['pe_ratio']:.2f}",
                            'ROE (%)': f"{metrics['roe'] * 100:.2f}%",
                            'Cổ Tức (%)': f"{metrics['dividend_yield'] * 100:.2f}%"
                        })
                    
                    df = pd.DataFrame(portfolio_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Create pie chart for portfolio allocation
                    fig = go.Figure(data=[go.Pie(
                        labels=[f"{row['Mã CK']} ({row['Tỷ Trọng (%)']})" for _, row in df.iterrows()],
                        values=[float(row['Tỷ Trọng (%)'].strip('%')) for _, row in df.iterrows()],
                        hole=.3
                    )])
                    fig.update_layout(title="Phân Bổ Danh Mục")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed analysis
                    st.subheader("Phân Tích Chi Tiết")
                    
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["So Sánh Điểm Số", "So Sánh Thời Gian Hoàn Vốn"])
                    
                    with tab1:
                        # Create bar chart for scores
                        fig = go.Figure(data=[
                            go.Bar(
                                x=df['Mã CK'],
                                y=[float(score.strip('/100')) for score in df['Điểm Số']],
                                text=df['Điểm Số'],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title="So Sánh Điểm Số Tài Chính",
                            yaxis_title="Điểm Số",
                            yaxis_range=[0, 100]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Create bar chart for payback times
                        fig = go.Figure(data=[
                            go.Bar(
                                x=df['Mã CK'],
                                y=[float(time.strip(' năm')) for time in df['Thời Gian HV']],
                                text=df['Thời Gian HV'],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title="So Sánh Thời Gian Hoàn Vốn",
                            yaxis_title="Năm"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Lỗi khi phân tích danh mục: {str(e)}")
                st.error("Vui lòng kiểm tra lại danh sách mã chứng khoán và thử lại.")
    elif menu_option == "Trợ Lý Đầu Tư":
        st.title("Trợ Lý Đầu Tư AI")
        st.write("Tương tác với AI để nhận tư vấn và phân tích về đầu tư chứng khoán")
        
        # Initialize session state for chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Xin chào! Tôi là trợ lý AI của bạn. Tôi có thể giúp bạn phân tích và tư vấn về đầu tư chứng khoán. Bạn có thể hỏi tôi về:\n\n"
                "1. Phân tích tài chính của một công ty\n"
                "2. Xu hướng giá và chỉ báo kỹ thuật\n"
                "3. Dự báo giá trong tương lai\n"
                "4. So sánh các cổ phiếu\n"
                "5. Tư vấn phân bổ danh mục\n\n"
                "Hãy nhập mã chứng khoán và câu hỏi của bạn!"
            })
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Get user input
        if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process the query
            try:
                with st.spinner("Đang xử lý..."):
                    # Extract stock symbol and financial metric from the prompt
                    common_metrics = {
                        'ROE', 'ROA', 'P/E', 'EPS', 'FCF', 'PE', 'PB', 'PS', 'EV', 'EBITDA', 
                        'NIM', 'CIR', 'CLR', 'LDR', 'LAR', 'ROE', 'ROA', 'P/E', 'EPS', 'FCF', 
                        'PE', 'PB', 'PS', 'EV', 'EBITDA', 'NIM', 'CIR', 'CLR', 'LDR', 'LAR'
                    }
                    
                    # First try to find a stock symbol (2-3 uppercase letters not in common_metrics)
                    symbols = [
                        word for word in prompt.split() 
                        if word.isupper() 
                        and len(word) <= 3 
                        and word not in common_metrics
                    ]
                    symbol = symbols[0] if symbols else None
                    
                    # Then try to find a financial metric
                    metrics = [
                        word for word in prompt.split() 
                        if word.isupper() 
                        and word in common_metrics
                    ]
                    metric = metrics[0] if metrics else None
                    
                    # Extract time period if mentioned
                    time_period = None
                    if '5' in prompt:
                        time_period = 5
                    elif '10' in prompt:
                        time_period = 10
                    elif '3' in prompt:
                        time_period = 3
                    
                    # Prepare context for the AI
                    context = ""
                    if symbol and metric:
                        try:
                            # Initialize components
                            api_client = FireantAPI()
                            scorer = CompanyScorer(symbol, api_client)
                            
                            # Get the metric value
                            if metric == 'ROE':
                                value = scorer.data['latest'].get('ROE', 0) * 100
                            elif metric == 'ROA':
                                value = scorer.data['latest'].get('NetProfitFromOperatingActivity', 0) / scorer.data['latest'].get('TotalAsset', 1) * 100
                            elif metric == 'P/E':
                                value = scorer.data['fundamental']['financial_metrics']['P/E']
                            elif metric == 'EPS':
                                value = scorer.data['fundamental']['financial_metrics']['EPS']
                            elif metric == 'FCF':
                                value = scorer.data['latest'].get('FreeCashFlow', 0)
                            else:
                                value = 0
                            
                            context = f"""
                            Thông tin {metric} của {symbol}:
                            - Giá trị hiện tại: {value:.2f}
                            - Công ty: {scorer.data['fundamental']['company_info']['Tên công ty']}
                            - Sàn: {scorer.data['fundamental']['company_info']['Sàn']}
                            """
                        except Exception as e:
                            context = f"Không thể lấy dữ liệu {metric} cho {symbol}. Lỗi: {str(e)}"
                    
                    # Prepare the prompt for the AI
                    ai_prompt = f"""
                    Bạn là một trợ lý AI chuyên về phân tích đầu tư chứng khoán. 
                    Hãy trả lời câu hỏi sau của người dùng một cách chuyên nghiệp và hữu ích.
                    
                    {context}
                    
                    Câu hỏi của người dùng: {prompt}
                    
                    Hãy trả lời bằng tiếng Việt và đảm bảo:
                    1. Sử dụng ngôn ngữ dễ hiểu
                    2. Cung cấp thông tin chi tiết và có căn cứ
                    3. Đưa ra khuyến nghị rõ ràng nếu được yêu cầu
                    4. Luôn nhắc nhở về rủi ro đầu tư
                    """
                    
                    # Get AI response
                    response = model.generate_content(ai_prompt)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                        
            except Exception as e:
                error_message = f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.markdown(error_message)

if __name__ == "__main__":
    main() 