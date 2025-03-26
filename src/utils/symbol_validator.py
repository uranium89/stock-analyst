"""
Symbol validation using Gemini AI
"""

import google.generativeai as genai
import json
from typing import Dict, Any
from src.config.settings import GOOGLE_API_KEY, GEMINI_MODEL

class SymbolValidator:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def validate_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Validate if a given symbol is a valid stock symbol on the Vietnamese stock market
        """
        prompt = f"""
        Hãy kiểm tra xem mã chứng khoán "{symbol}" có phải là mã chứng khoán hợp lệ trên thị trường chứng khoán Việt Nam hay không.
        
        Yêu cầu:
        1. Mã chứng khoán phải là mã của công ty niêm yết trên sàn HOSE hoặc HNX
        2. Mã chứng khoán phải có độ dài từ 2-3 ký tự
        3. Mã chứng khoán phải chỉ chứa chữ cái in hoa
        4. Mã chứng khoán phải là mã đang được giao dịch (không phải mã đã ngừng giao dịch)
        5. Mã chứng khoán KHÔNG được là các thuật ngữ tài chính phổ biến như ROE, ROA, P/E, EPS, FCF, PE, PB, PS, EV, EBITDA, NIM, CIR, CLR, LDR, LAR
        
        Ví dụ hợp lệ: VNM, VCB, VHM, FPT, VIC
        Ví dụ không hợp lệ: ROE, ROA, P/E, EPS
        
        Hãy trả lời theo định dạng JSON chính xác:
        {{
            "is_valid": true/false,
            "reason": "Lý do nếu không hợp lệ",
            "exchange": "HOSE/HNX nếu hợp lệ",
            "company_name": "Tên công ty nếu hợp lệ"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove any markdown code block markers if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            # Parse the JSON response
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            return {
                "is_valid": False,
                "reason": f"Lỗi khi xử lý phản hồi từ AI: {str(e)}",
                "exchange": None,
                "company_name": None
            }
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Lỗi khi kiểm tra mã: {str(e)}",
                "exchange": None,
                "company_name": None
            } 