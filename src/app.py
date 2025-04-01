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
from src.charts.chart_creator import ChartCreator
from src.utils.api_client import FireantAPI
from src.predictors.ai_price_predictor import AIPricePredictor

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
            ["Phân Tích Cổ Phiếu", "Phân Bổ Danh Mục", "Phân Tích Thị Trường"]
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
                    tab1, tab2, tab3 = st.tabs(["Phân Tích Tài Chính", "Phân Tích Xu Hướng Giá", "Báo Cáo Chi Tiết"])
                    
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
                            # Calculate ROA with fallback options
                            try:
                                # Print available columns for debugging
                                print(f"Available financial columns: {list(latest.keys())}")
                                
                                # Try different possible column names for operating profit
                                profit_columns = [
                                    'NetProfitFromOperatingActivity',
                                    'OperatingProfit',
                                    'OperatingIncome',
                                    'OperatingActivities',
                                    'OperatingCashFlow',
                                    'CashFlowFromOperatingActivities',
                                    'OperatingActivitiesCashFlow',
                                    'CashFlowFromOperations',
                                    'NetIncome',  # Fallback to net income if operating profit not available
                                    'ProfitAfterTax'  # Another common name for net income
                                ]
                                
                                operating_profit = 0
                                for col in profit_columns:
                                    if col in latest:
                                        operating_profit = latest[col]
                                        print(f"Found profit column: {col}")
                                        break
                                
                                if operating_profit == 0:
                                    print("Warning: No profit data found in available columns")
                                
                                total_assets = latest.get('TotalAsset', 1)
                                roa = (operating_profit / total_assets) * 100 if total_assets > 0 else 0
                                
                                st.metric(
                                    "ROA",
                                    f"{roa:.2f}%",
                                    f"{roa - 5:.2f}%"
                                )
                            except Exception as e:
                                print(f"Warning: Could not calculate ROA: {str(e)}")
                                st.metric(
                                    "ROA",
                                    "N/A",
                                    "Không có dữ liệu"
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
                        # Display AI Prediction with detailed error handling
                        st.subheader("Dự Đoán AI")
                        try:
                            if 'ai_prediction' in price_analysis:
                                ai_pred = price_analysis['ai_prediction']
                                
                                # Create columns for AI metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Xu Hướng Dự Đoán",
                                        ai_pred['trend'],
                                        f"{ai_pred['predicted_return']:.2%}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Độ Tin Cậy",
                                        f"{ai_pred['confidence']:.2%}",
                                        "Cao" if ai_pred['confidence'] > 0.02 else "Trung Bình" if ai_pred['confidence'] > 0.01 else "Thấp"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Sức Mạnh Tín Hiệu",
                                        ai_pred['strength'],
                                        f"R²: {ai_pred['model_performance']['test_score']:.2f}"
                                    )
                                
                                with col4:
                                    st.metric(
                                        "Độ Chính Xác",
                                        f"{ai_pred['model_performance']['test_score']:.2%}",
                                        f"{ai_pred['model_performance']['test_score'] - 0.5:.2%}"
                                    )
                                
                                # Display top features
                                st.subheader("Các Chỉ Số Quan Trọng Nhất")
                                feature_cols = st.columns(5)
                                for i, (feature, importance) in enumerate(list(ai_pred['top_features'].items())[:5]):
                                    with feature_cols[i]:
                                        st.metric(
                                            feature,
                                            f"{importance:.2f}",
                                            "Quan trọng" if importance > 0.1 else "Trung bình" if importance > 0.05 else "Thấp"
                                        )
                            else:
                                st.warning("Không tìm thấy dự đoán AI trong kết quả phân tích.")
                                st.info("Đang thử tạo dự đoán AI...")
                                
                                # Try to create AI prediction directly
                                ai_metrics = price_analyzer.ai_predictor.train()
                                ai_prediction = price_analyzer.ai_predictor.predict()
                                ai_importance = price_analyzer.ai_predictor.get_feature_importance()
                                
                                # Update price_analysis with AI prediction
                                price_analysis['ai_prediction'] = {
                                    'trend': ai_prediction['trend'],
                                    'predicted_return': ai_prediction['predicted_return'],
                                    'confidence': ai_prediction['confidence'],
                                    'strength': ai_prediction['strength'],
                                    'model_performance': ai_metrics,
                                    'top_features': dict(list(ai_importance.items())[:5])
                                }
                                
                                # Display the prediction
                                ai_pred = price_analysis['ai_prediction']
                                
                                # Create columns for AI metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Xu Hướng Dự Đoán",
                                        ai_pred['trend'],
                                        f"{ai_pred['predicted_return']:.2%}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Độ Tin Cậy",
                                        f"{ai_pred['confidence']:.2%}",
                                        "Cao" if ai_pred['confidence'] > 0.02 else "Trung Bình" if ai_pred['confidence'] > 0.01 else "Thấp"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Sức Mạnh Tín Hiệu",
                                        ai_pred['strength'],
                                        f"R²: {ai_pred['model_performance']['test_score']:.2f}"
                                    )
                                
                                with col4:
                                    st.metric(
                                        "Độ Chính Xác",
                                        f"{ai_pred['model_performance']['test_score']:.2%}",
                                        f"{ai_pred['model_performance']['test_score'] - 0.5:.2%}"
                                    )
                                
                                # Display top features
                                st.subheader("Các Chỉ Số Quan Trọng Nhất")
                                feature_cols = st.columns(5)
                                for i, (feature, importance) in enumerate(list(ai_pred['top_features'].items())[:5]):
                                    with feature_cols[i]:
                                        st.metric(
                                            feature,
                                            f"{importance:.2f}",
                                            "Quan trọng" if importance > 0.1 else "Trung bình" if importance > 0.05 else "Thấp"
                                        )
                        except Exception as e:
                            st.error(f"Lỗi khi tạo dự đoán AI: {str(e)}")
                            st.error("Chi tiết lỗi:")
                            st.exception(e)
                            st.info("Vui lòng thử lại sau hoặc liên hệ hỗ trợ nếu lỗi vẫn tiếp tục.")
                        
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
    elif menu_option == "Phân Tích Thị Trường":
        st.title("Phân Tích Thị Trường")
        st.write("Phân tích xu hướng và chỉ báo kỹ thuật của thị trường chung")
        
        # Initialize components
        api_client = FireantAPI()
        chart_creator = ChartCreator()
        report_generator = ReportGenerator()
        
        try:
            with st.spinner("Đang phân tích dữ liệu thị trường..."):
                # Get VNINDEX data for the last year
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.DateOffset(years=1)
                
                # Fetch VNINDEX data
                vnindex_data = api_client.get_historical_quotes(
                    "VNINDEX",
                    start_date.strftime('%m/%d/%Y'),
                    end_date.strftime('%m/%d/%Y')
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(vnindex_data)
                df['Date'] = pd.to_datetime(df['date'])
                df = df.sort_values('Date')
                
                # Calculate technical indicators
                price_analyzer = PriceTrendAnalyzer("VNINDEX")
                df_with_indicators = price_analyzer.calculate_technical_indicators()
                
                # Get AI prediction
                ai_predictor = AIPricePredictor("VNINDEX")
                ai_metrics = ai_predictor.train()
                ai_prediction = ai_predictor.predict()
                ai_importance = ai_predictor.get_feature_importance()
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Phân Tích Kỹ Thuật", "Dự Đoán AI", "Báo Cáo Chi Tiết"])
                
                with tab1:
                    # Display market overview
                    st.subheader("Tổng Quan Thị Trường")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    latest = df_with_indicators.iloc[-1]
                    prev = df_with_indicators.iloc[-2]
                    
                    with col1:
                        st.metric(
                            "VNINDEX",
                            f"{latest['priceClose']:,.2f}",
                            f"{((latest['priceClose'] - prev['priceClose']) / prev['priceClose'] * 100):.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "RSI",
                            f"{latest['RSI']:.2f}",
                            "Quá mua" if latest['RSI'] > 70 else "Quá bán" if latest['RSI'] < 30 else "Trung tính"
                        )
                    
                    with col3:
                        st.metric(
                            "MACD",
                            f"{latest['MACD']:.2f}",
                            "Mua" if latest['MACD'] > latest['Signal_Line'] else "Bán"
                        )
                    
                    with col4:
                        st.metric(
                            "Xu Hướng",
                            "Tăng" if latest['priceClose'] > latest['MA20'] else "Giảm",
                            "Ngắn hạn"
                        )
                    
                    # Display price trend chart
                    st.subheader("Biểu Đồ Xu Hướng Giá")
                    st.plotly_chart(
                        chart_creator.create_price_trend_chart(df_with_indicators),
                        use_container_width=True
                    )
                    
                    # Display technical indicators
                    st.subheader("Chỉ Báo Kỹ Thuật")
                    st.plotly_chart(
                        chart_creator.create_technical_indicators_chart(df_with_indicators),
                        use_container_width=True
                    )
                    
                    # Market analysis
                    st.subheader("Phân Tích Thị Trường")
                    
                    # Calculate market trend
                    ma20 = df_with_indicators['MA20'].iloc[-1]
                    ma50 = df_with_indicators['MA50'].iloc[-1]
                    ma200 = df_with_indicators['MA200'].iloc[-1]
                    current_price = df_with_indicators['priceClose'].iloc[-1]
                    
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
                    
                    # Display market analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Xu Hướng Thị Trường")
                        st.write(f"- **Xu hướng hiện tại:** {trend}")
                        st.write(f"- **RSI:** {latest['RSI']:.2f} ({'Quá mua' if latest['RSI'] > 70 else 'Quá bán' if latest['RSI'] < 30 else 'Trung tính'})")
                        st.write(f"- **MACD:** {'Tín hiệu mua' if latest['MACD'] > latest['Signal_Line'] else 'Tín hiệu bán'}")
                    
                    with col2:
                        st.markdown("##### Chỉ Báo Kỹ Thuật")
                        st.write(f"- **MA20:** {ma20:,.2f}")
                        st.write(f"- **MA50:** {ma50:,.2f}")
                        st.write(f"- **MA200:** {ma200:,.2f}")
                        st.write(f"- **Bollinger Bands:** {'Quá mua' if current_price > latest['BB_upper'] else 'Quá bán' if current_price < latest['BB_lower'] else 'Trung tính'}")
                    
                    # Volume analysis
                    st.subheader("Phân Tích Khối Lượng")
                    volume_ma20 = df_with_indicators['dealVolume'].rolling(window=20).mean().iloc[-1]
                    current_volume = df_with_indicators['dealVolume'].iloc[-1]
                    
                    st.write(f"- **Khối lượng hiện tại:** {current_volume:,.0f}")
                    st.write(f"- **Khối lượng trung bình 20 phiên:** {volume_ma20:,.0f}")
                    st.write(f"- **So sánh:** {'Cao hơn' if current_volume > volume_ma20 else 'Thấp hơn'} trung bình")
                
                with tab2:
                    # Display AI Prediction
                    st.subheader("Dự Đoán AI")
                    
                    # Create columns for AI metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Xu Hướng Dự Đoán",
                            ai_prediction['trend'],
                            f"{ai_prediction['predicted_return']:.2%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Độ Tin Cậy",
                            f"{ai_prediction['confidence']:.2%}",
                            "Cao" if ai_prediction['confidence'] > 0.02 else "Trung Bình" if ai_prediction['confidence'] > 0.01 else "Thấp"
                        )
                    
                    with col3:
                        st.metric(
                            "Sức Mạnh Tín Hiệu",
                            ai_prediction['strength'],
                            f"R²: {ai_metrics['test_score']:.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Độ Chính Xác",
                            f"{ai_metrics['test_score']:.2%}",
                            f"{ai_metrics['test_score'] - 0.5:.2%}"
                        )
                    
                    # Display top features
                    st.subheader("Các Chỉ Số Quan Trọng Nhất")
                    feature_cols = st.columns(5)
                    for i, (feature, importance) in enumerate(list(ai_importance.items())[:5]):
                        with feature_cols[i]:
                            st.metric(
                                feature,
                                f"{importance:.2f}",
                                "Quan trọng" if importance > 0.1 else "Trung bình" if importance > 0.05 else "Thấp"
                            )
                
                with tab3:
                    # Add a button to generate reports
                    if st.button("Tạo Báo Cáo Chi Tiết"):
                        with st.spinner("Đang tạo báo cáo chi tiết..."):
                            # Generate market analysis report
                            market_report = report_generator.generate_market_report(
                                df_with_indicators,
                                ai_prediction,
                                ai_metrics,
                                ai_importance
                            )
                            
                            # Display detailed report
                            st.subheader("Báo Cáo Phân Tích Thị Trường")
                            st.write(market_report)
                    else:
                        st.info("Nhấn nút 'Tạo Báo Cáo Chi Tiết' để xem phân tích chi tiết được tạo bởi AI.")
                
        except Exception as e:
            st.error(f"Lỗi khi phân tích thị trường: {str(e)}")
            st.error("Vui lòng thử lại sau.")
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