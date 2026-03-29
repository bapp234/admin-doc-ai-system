import re

def is_garbage_text(text: str) -> bool:
    text = text.strip()
    if not text: return True
    
    if len(text) <= 2 and not text.isdigit() and text.upper() not in ["V", "Y", "A", "B", "C"]:
        return True
        
    special_chars = re.findall(r'[^a-zA-Z0-9\s_a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỮỰỲỴÝỶỸửữựỳỵỷỹ]', text)
    if len(special_chars) > len(text) * 0.35:
        return True
        
    words = text.split()
    longest_word = max(words, key=len) if words else ""
    if len(longest_word) > 8 and not longest_word.isdigit() and not any(c.isdigit() for c in longest_word):
        if "@" not in text and "http" not in text:
            return True

    return False

def clean_vba_text(text: str) -> str:
    if is_garbage_text(text): return ""
        
    text = text.strip()
    replacements = {
        "11DA": "HÒA", "11OA": "HÒA", "TỈNII": "TỈNH", "11ghiệp": "nghiệp",
        "PINT": "PTNT", "I lanh": "Hạnh", "ILạnh": "Hạnh", "phíc": "phúc",
        "UBNI)": "UBND", "URVD": "UBND", "I|": "H", "I-I": "H", "CÔNG HÒA": "CỘNG HÒA"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r'(\d{1,2})[\s\.\/I](\d{1,2})[\s\.\/I](\d{4})', r'\1/\2/\3', text)