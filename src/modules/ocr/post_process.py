import re

def is_garbage_text(text):
    """Phát hiện và tiêu diệt các đoạn text rác sinh ra từ Con dấu / Chữ ký"""
    text = text.strip()
    if not text: return True
    
    # 1. Rác nhiễu (Chỉ có 1-2 ký tự vô nghĩa)
    if len(text) <= 2 and not text.isdigit() and text.upper() not in ["V", "Y", "A", "B", "C"]:
        return True
        
    # 2. Rác con dấu (Chứa quá nhiều ký tự đặc biệt như đường viền, hoa văn)
    # Lọc ra các ký tự KHÔNG phải là chữ cái Tiếng Việt hoặc số
    special_chars = re.findall(r'[^a-zA-Z0-9\s_a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỮỰỲỴÝỶỸửữựỳỵỷỹ]', text)
    if len(special_chars) > len(text) * 0.35: # Hơn 35% là ký tự lạ -> Chắc chắn là con dấu
        return True
        
    # 3. Rác chữ ký (Tạo ra các chuỗi dài ngoẵng không có khoảng trắng như "Ghoouch")
    words = text.split()
    longest_word = max(words, key=len) if words else ""
    if len(longest_word) > 8 and not longest_word.isdigit() and not any(c.isdigit() for c in longest_word):
        if "@" not in text and "http" not in text: # Loại trừ trường hợp Email/Website thực sự
            return True

    return False

def clean_vba_text(text):
    """Hậu xử lý văn bản hành chính (VBA - Administrative Documents)."""
    # 🔴 CHẶN ĐỨNG RÁC NGAY TỪ CỬA
    if is_garbage_text(text):
        return ""
        
    text = text.strip()
    
    replacements = {
        "11DA": "HÒA", "11OA": "HÒA", "TỈNII": "TỈNH", "11ghiệp": "nghiệp",
        "PINT": "PTNT", "I lanh": "Hạnh", "ILạnh": "Hạnh", "phíc": "phúc",
        "UBNI)": "UBND", "URVD": "UBND", "I|": "H", "I-I": "H",
        "CÔNG HÒA": "CỘNG HÒA"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)

    date_pattern = r'(\d{1,2})[\s\.\/I](\d{1,2})[\s\.\/I](\d{4})'
    text = re.sub(date_pattern, r'\1/\2/\3', text)
    
    return text