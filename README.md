# AI Trading Analysis

Ứng dụng phân tích tài chính và xu hướng giá tự động sử dụng AI cho thị trường chứng khoán Việt Nam.

## Tính Năng

- Phân tích tài chính toàn diện cho doanh nghiệp và ngân hàng
- Phân tích xu hướng giá và chỉ báo kỹ thuật
- Tạo báo cáo phân tích chi tiết bằng AI
- Hiển thị trực quan với các biểu đồ tương tác
- Tính toán điểm số và khuyến nghị đầu tư

## Cài Đặt

1. Clone repository:
```bash
git clone https://github.com/yourusername/ai-trading.git
cd ai-trading
```

2. Tạo môi trường ảo và kích hoạt:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

4. Tạo file .env và thêm các biến môi trường:
```
GOOGLE_API_KEY=your_google_api_key
FIREANT_TOKEN=your_fireant_token
```

## Sử Dụng

1. Chạy ứng dụng:
```bash
streamlit run src/app.py
```

2. Mở trình duyệt và truy cập địa chỉ được hiển thị (thường là http://localhost:8501)

3. Nhập mã chứng khoán cần phân tích (VD: VNM, VCB, VHM)

## Cấu Trúc Dự Án

```
ai-trading/
├── src/
│   ├── analyzers/
│   │   ├── company_scorer.py
│   │   ├── price_analyzer.py
│   │   └── report_generator.py
│   ├── charts/
│   │   └── chart_creator.py
│   ├── config/
│   │   └── settings.py
│   ├── utils/
│   │   └── api_client.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Các Thành Phần Chính

### CompanyScorer
- Phân tích tài chính và tính điểm cho doanh nghiệp
- Xử lý dữ liệu tài chính và tính toán các chỉ số quan trọng
- Phân biệt và xử lý riêng cho ngân hàng và doanh nghiệp thông thường

### PriceTrendAnalyzer
- Phân tích xu hướng giá và chỉ báo kỹ thuật
- Tính toán các chỉ báo như RSI, MACD, Bollinger Bands
- Đánh giá xu hướng giá ngắn hạn và dài hạn

### ReportGenerator
- Tạo báo cáo phân tích chi tiết sử dụng AI
- Tổng hợp và trình bày kết quả phân tích
- Đưa ra khuyến nghị đầu tư

### ChartCreator
- Tạo các biểu đồ trực quan
- Hiển thị dữ liệu tài chính và kỹ thuật
- Tương tác và tùy chỉnh biểu đồ

## Đóng Góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit các thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## Giấy Phép

Dự án này được phân phối dưới giấy phép MIT. Xem `LICENSE` để biết thêm thông tin.

## Liên Hệ

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Link dự án: [https://github.com/yourusername/ai-trading](https://github.com/yourusername/ai-trading) 