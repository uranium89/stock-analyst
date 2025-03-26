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
            ["Phân Tích Cổ Phiếu", "Phân Bổ Danh Mục"]
        )
    
    if menu_option == "Phân Tích Cổ Phiếu":
        st.title("AI Trading Analysis")
        st.write("Phân tích tài chính và xu hướng giá tự động sử dụng AI")
        
        # Initialize components
        api_client = FireantAPI()
        chart_creator = ChartCreator()
        report_generator = ReportGenerator()
        
        # Get user input
        symbol = st.text_input("Nhập mã chứng khoán (VD: VNM, VCB, VHM):", "").upper()
        
        if symbol:
            try:
                with st.spinner("Đang phân tích dữ liệu..."):
                    # Initialize analyzers
                    scorer = CompanyScorer(symbol)
                    price_analyzer = PriceTrendAnalyzer(symbol)
                    
                    # Get analysis results
                    financial_result = scorer.calculate_score()
                    price_analysis = price_analyzer.analyze_trend()
                    price_data = price_analyzer.price_data
                    
                    # Update title with company name
                    company_name = scorer.data['fundamental']['company_info']['Tên công ty']
                    st.title(f"Phân Tích Điểm Số Tài Chính - {company_name} ({symbol})")
                    
                    st.markdown(f"**Loại Công Ty:** {'Ngân Hàng' if scorer.company_type == 'bank' else 'Công Ty Thường'}")
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Phân Tích Tài Chính", "Phân Tích Xu Hướng Giá", "Dự Đoán Giá AI", "Báo Cáo Chi Tiết"])
                    
                    with tab1:
                        # Display key financial metrics
                        st.subheader("Chỉ Số Tài Chính Chính")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "P/E",
                                f"{scorer.data['fundamental']['financial_metrics']['P/E']:.2f}",
                                f"{scorer.data['fundamental']['financial_metrics']['P/E'] - 15:.2f}"
                            )
                            st.metric(
                                "EPS",
                                f"{scorer.data['fundamental']['financial_metrics']['EPS']:.2f}",
                                f"{scorer.data['fundamental']['financial_metrics']['EPS'] - 1:.2f}"
                            )
                            st.metric(
                                "Giá Hiện Tại",
                                f"{scorer.data['fundamental']['company_info']['Giá hiện tại']:,.0f}",
                                f"{scorer.data['fundamental']['company_info']['Giá hiện tại'] - 10000:,.0f}"
                            )
                        
                        with col2:
                            latest = scorer.data['latest']
                            st.metric(
                                "ROE",
                                f"{latest.get('ROE', 0) * 100:.2f}%",
                                f"{latest.get('ROE', 0) * 100 - 15:.2f}%"
                            )
                            st.metric(
                                "ROA",
                                f"{latest.get('NetProfitFromOperatingActivity', 0) / latest.get('TotalAsset', 1) * 100:.2f}%",
                                f"{latest.get('NetProfitFromOperatingActivity', 0) / latest.get('TotalAsset', 1) * 100 - 5:.2f}%"
                            )
                            payback = scorer.payback_analysis
                            st.metric(
                                "Thời Gian Hoàn Vốn",
                                f"{payback['payback_years']['moderate']:.1f} năm",
                                f"{payback['payback_years']['moderate'] - 5:.1f} năm"
                            )
                            st.metric(
                                "Cổ Tức Trung Bình",
                                f"{payback['metrics']['avg_dividend']:,.0f}",
                                f"{payback['metrics']['avg_dividend_growth']:.1%}"
                            )
                        
                        with col3:
                            latest = scorer.data['latest']
                            st.metric(
                                "Cổ Tức (%)",
                                f"{latest.get('DividendYield', 0) * 100:.2f}%",
                                f"{latest.get('DividendYield', 0) * 100 - 5:.2f}%"
                            )
                            st.metric(
                                "Vốn Hóa",
                                f"{scorer.data['fundamental']['company_info']['Vốn hóa']:,.0f}",
                                f"{scorer.data['fundamental']['company_info']['Vốn hóa'] - 1000000000:,.0f}"
                            )
                            st.metric(
                                "Mức Độ Ổn Định Cổ Tức",
                                scorer.payback_analysis['dividend_stability'],
                                f"{scorer.payback_analysis['metrics']['avg_coverage_ratio']:.2f}"
                            )
                        
                        with col4:
                            company_info = scorer.data['fundamental']['company_info']
                            st.metric(
                                "Beta",
                                f"{company_info.get('Beta', 0):.2f}",
                                f"{company_info.get('Beta', 0) - 1:.2f}"
                            )
                            # Try different possible keys for foreign ownership room and convert to percentage
                            foreign_room = (
                                company_info.get('Room nước ngoài', 0) or 
                                company_info.get('foreignPercent', 0) or 
                                company_info.get('foreignOwnership', 0) or 
                                company_info.get('foreign_ownership', 0)
                            )
                            # Convert to percentage if not already in percentage form
                            if foreign_room > 1:
                                foreign_room = foreign_room / 100
                            st.metric(
                                "Room NN",
                                f"{foreign_room * 100:.2f}%",
                                f"{foreign_room * 100 - 30:.2f}%"
                            )
                        
                        # Display financial score
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(
                                chart_creator.create_gauge_chart(
                                    financial_result['final_score'],
                                    f"Điểm Số Tài Chính - {symbol}"
                                ),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.plotly_chart(
                                chart_creator.create_trend_chart(
                                    financial_result['business_cycle']['yearly_revenue_growth'],
                                    f"Tăng Trưởng Doanh Thu - {symbol}",
                                    "Tăng Trưởng (%)"
                                ),
                                use_container_width=True
                            )
                        
                        # Display ROE and Equity charts
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(
                                chart_creator.create_roe_chart(
                                    scorer.data['yearly']
                                ),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.plotly_chart(
                                chart_creator.create_equity_chart(
                                    scorer.data['yearly']
                                ),
                                use_container_width=True
                            )
                        
                        # Display dividend chart
                        st.plotly_chart(
                            chart_creator.create_dividend_chart(
                                scorer.data['yearly']
                            ),
                            use_container_width=True
                        )
                    
                    with tab2:
                        # Calculate technical indicators first
                        price_data_with_indicators = price_analyzer.calculate_technical_indicators()
                        
                        # Display price trend chart
                        st.plotly_chart(
                            chart_creator.create_price_trend_chart(price_data_with_indicators),
                            use_container_width=True
                        )
                        
                        # Display technical indicators
                        st.plotly_chart(
                            chart_creator.create_technical_indicators_chart(price_data_with_indicators),
                            use_container_width=True
                        )
                        
                        # Display price analysis score
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(
                                chart_creator.create_gauge_chart(
                                    price_analysis['final_score'],
                                    f"Điểm Số Xu Hướng Giá - {symbol}"
                                ),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.metric(
                                "Điểm Số Xu Hướng",
                                f"{price_analysis['final_score']:.2f}/100",
                                f"{price_analysis['final_score'] - 50:.2f}"
                            )
                    
                    with tab3:
                        st.subheader("Dự Đoán Giá Sử Dụng AI")
                        st.write("Mô hình AI được sử dụng để dự đoán xu hướng giá trong tương lai")
                        
                        if st.button("Chạy Dự Đoán"):
                            logger.info(f"Price prediction button clicked for symbol: {symbol}")
                            try:
                                with st.spinner("Đang lấy dữ liệu giá..."):
                                    # Fetch historical price data from API
                                    end_date = datetime.now()
                                    days=365
                                    start_date = end_date - timedelta(days=days)  # 1 years
                                    start_date_str = start_date.strftime('%m/%d/%Y')
                                    end_date_str = end_date.strftime('%m/%d/%Y')
                                    
                                    logger.info(f"Fetching price data for {symbol} from {start_date_str} to {end_date_str}")
                                    price_data = api_client.get_historical_quotes(symbol, start_date_str, end_date_str, offset=0, limit=days)
                                    
                                    if not price_data:
                                        st.error("Không thể lấy dữ liệu giá từ API. Vui lòng thử lại sau.")
                                        return
                                        
                                    # Convert to DataFrame
                                    price_df = pd.DataFrame(price_data)
                                    price_df['date'] = pd.to_datetime(price_df['date'])
                                    price_df.set_index('date', inplace=True)
                                    
                                    if 'priceClose' not in price_df.columns:
                                        st.error("Dữ liệu giá không có cột 'priceClose'. Vui lòng thử lại sau.")
                                        return
                                        
                                    price_df = price_df.sort_index()
                                    
                                    logger.info(f"Got {len(price_df)} days of price data")
                                    
                                    if len(price_df) < 60:
                                        st.error(f"Không đủ dữ liệu để thực hiện dự đoán. Cần ít nhất 60 ngày dữ liệu giá, hiện có {len(price_df)} ngày.")
                                        return
                                        
                                with st.spinner("Đang huấn luyện mô hình AI..."):
                                    logger.info("Initializing PricePredictor")
                                    # Initialize and train the model
                                    predictor = PricePredictor()
                                    logger.info("Training model...")
                                    prediction_result = predictor.train(price_df, symbol)
                                    
                                    logger.info("Getting next days predictions")
                                    # Get predictions for next 5 days
                                    next_days_predictions = predictor.predict_next_days(price_df, symbol)
                                    
                                    # Display current prediction
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            "Giá Dự Đoán Ngày Mai",
                                            f"{prediction_result['predicted_price']:,.0f}",
                                            f"{prediction_result['predicted_price'] - price_df['priceClose'].iloc[-1]:,.0f}"
                                        )
                                    
                                    with col2:
                                        avg_score = (prediction_result['model_metrics']['rf_score'] + prediction_result['model_metrics']['xgb_score']) / 2
                                        st.metric(
                                            "Độ Tin Cậy Mô Hình",
                                            f"{avg_score:.2%}",
                                            f"{prediction_result['model_metrics']['rf_score']:.2%} - {prediction_result['model_metrics']['xgb_score']:.2%}"
                                        )
                                    
                                    # Create prediction chart
                                    fig = go.Figure()
                                    
                                    # Add historical prices
                                    fig.add_trace(go.Scatter(
                                        x=price_df.index,
                                        y=price_df['priceClose'],
                                        name='Giá Lịch Sử',
                                        line=dict(color='blue')
                                    ))
                                    
                                    # Add predicted prices
                                    future_dates = pd.date_range(
                                        start=price_df.index[-1] + pd.Timedelta(days=1),
                                        periods=len(next_days_predictions),
                                        freq='D'
                                    )
                                    
                                    predicted_prices = [p['predicted_price'] for p in next_days_predictions]
                                    confidence_lower = [p['confidence_interval']['lower'] for p in next_days_predictions]
                                    confidence_upper = [p['confidence_interval']['upper'] for p in next_days_predictions]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=future_dates,
                                        y=predicted_prices,
                                        name='Dự Đoán',
                                        line=dict(color='red', dash='dash')
                                    ))
                                    
                                    # Add confidence interval
                                    fig.add_trace(go.Scatter(
                                        x=future_dates.tolist() + future_dates.tolist()[::-1],
                                        y=confidence_upper + confidence_lower[::-1],
                                        fill='toself',
                                        fillcolor='rgba(255,0,0,0.1)',
                                        line=dict(color='rgba(255,0,0,0)'),
                                        name='Khoảng Tin Cậy'
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"Dự Đoán Giá {symbol}",
                                        xaxis_title="Ngày",
                                        yaxis_title="Giá",
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display detailed predictions
                                    st.subheader("Dự Đoán Chi Tiết")
                                    prediction_df = pd.DataFrame([
                                        {
                                            'Ngày': future_dates[i],
                                            'Giá Dự Đoán': f"{p['predicted_price']:,.0f}",
                                            'Khoảng Tin Cậy': f"{p['confidence_interval']['lower']:,.0f} - {p['confidence_interval']['upper']:,.0f}"
                                        }
                                        for i, p in enumerate(next_days_predictions)
                                    ])
                                    st.dataframe(prediction_df, use_container_width=True)
                                    
                            except ValueError as e:
                                logger.error(f"ValueError in price prediction: {str(e)}")
                                st.error(f"Lỗi khi dự đoán giá: {str(e)}")
                                st.info("Vui lòng đảm bảo có đủ dữ liệu giá (ít nhất 60 ngày) và thử lại.")
                            except Exception as e:
                                logger.error(f"Unexpected error in price prediction: {str(e)}")
                                st.error(f"Lỗi không mong muốn: {str(e)}")
                                st.info("Vui lòng thử lại sau.")
                    
                    with tab4:
                        # Add a button to generate reports
                        if st.button("Tạo Báo Cáo Chi Tiết"):
                            with st.spinner("Đang tạo báo cáo chi tiết..."):
                                financial_report = report_generator.generate_financial_report(scorer, financial_result)
                                price_report = report_generator.generate_price_analysis_report(price_analysis, price_data)
                                
                                # Display detailed reports
                                st.subheader("Báo Cáo Phân Tích Tài Chính")
                                st.write(financial_report)
                                
                                st.subheader("Báo Cáo Phân Tích Xu Hướng Giá")
                                st.write(price_report)
                        else:
                            st.info("Nhấn nút 'Tạo Báo Cáo Chi Tiết' để xem phân tích chi tiết được tạo bởi AI.")
            
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
    elif menu_option == "So Sánh Cổ Phiếu":
        st.title("So Sánh Cổ Phiếu")
        st.write("So sánh các chỉ số tài chính giữa các cổ phiếu")
        
        # Placeholder for stock comparison feature
        st.info("Tính năng so sánh cổ phiếu đang được phát triển. Vui lòng quay lại sau!")
        
    elif menu_option == "Cài Đặt":
        st.title("Cài Đặt")
        st.write("Tùy chỉnh các thông số phân tích")
        
        # Placeholder for settings
        st.info("Tính năng cài đặt đang được phát triển. Vui lòng quay lại sau!")

if __name__ == "__main__":
    main() 