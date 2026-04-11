# Hướng dẫn cài đặt (INSTALL)

Tài liệu này tổng hợp các bước để cài đặt các dependency cần thiết cho dự án `admin-doc-ai-system` trên Windows / Linux / macOS, bao gồm cả dependency hệ thống Poppler (binaries `pdftoppm` / `pdfinfo`).

---

## 1) Python + virtualenv (khuyến nghị)
- Khuyến nghị: Python 3.8 - 3.11

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Lưu ý: trên Colab bạn chỉ cần `pip install -r requirements.txt` trong một cell.

---

## 2) Poppler (binaries) — Bắt buộc cho xử lý PDF → ảnh
`pdf2image` và các công cụ chuyển PDF cần `pdftoppm` / `pdfinfo` từ Poppler. Poppler là dependency hệ thống (không phải pip).

### Kiểm tra đã cài chưa
- PowerShell / CMD (Windows):
```powershell
Get-Command pdftoppm -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
Get-Command pdfinfo -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
```
- Linux/macOS:
```bash
which pdftoppm || command -v pdftoppm
```
- Python quick-check:
```bash
python -c "import shutil; print(shutil.which('pdftoppm'), shutil.which('pdfinfo'))"
```

Nếu trả về `None` / rỗng → chưa cài Poppler.

### Windows — Cách cài
1. Chocolatey (nếu bạn dùng Chocolatey):
```powershell
choco install poppler -y
```
2. Winget (Windows 10/11):
```powershell
winget install --id Poppler.Poppler -e
```
3. Hoặc tải tay (nếu không có choco/winget):
- Tải release phù hợp tại: https://github.com/oschwartz10612/poppler-windows/releases
- Giải nén, ví dụ `C:\poppler-25.12.0`
- Thêm `C:\poppler-25.12.0\Library\bin` vào `PATH` (tạm thời):
```powershell
$env:PATH += ';C:\poppler-25.12.0\Library\bin'
```
- Để kiểm tra:
```powershell
pdftoppm -v
pdfinfo -v
```

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y poppler-utils
```

### Fedora
```bash
sudo dnf install -y poppler-utils
```

### macOS (Homebrew)
```bash
brew install poppler
```

### Conda (nếu dùng conda)
```bash
conda install -c conda-forge poppler
```

---

## 3) Lưu ý cho `pdf2image`
Nếu bạn dùng `pdf2image.convert_from_path()` và Poppler không nằm trong `PATH`, truyền `poppler_path`:
```python
from pdf2image import convert_from_path
pages = convert_from_path('file.pdf', poppler_path=r'C:\poppler-25.12.0\Library\bin')
```

---

## 4) Cài Python packages
Tất cả packages Python của project được liệt kê trong `requirements.txt` ở gốc repo.

Cài bằng pip (đã active venv):
```bash
pip install -r requirements.txt
```

Nếu muốn dùng Conda, tạo env và cài pip packages:
```bash
conda create -n admin-doc-ai python=3.10 -y
conda activate admin-doc-ai
pip install -r requirements.txt
```

---

## 5) Kiểm tra tổng quát sau cài
```bash
python -c "import shutil; print('pdftoppm=', shutil.which('pdftoppm'), 'pdfinfo=', shutil.which('pdfinfo'))"
python -c "import pkgutil; print('pdf2image', pkgutil.find_loader('pdf2image') is not None)"
python -c "import paddleocr, vietocr, transformers; print('imports OK')"
```

---

## 6) Scripts hỗ trợ (repo)
Trong `scripts/` có sẵn các script mẫu để setup môi trường và hướng dẫn cài Poppler trên Windows và Linux/macOS. Bạn có thể dùng `scripts/prepare_env.sh` (Linux/macOS) hoặc `scripts/prepare_env_windows.ps1` (PowerShell) để cài nhanh phần Python packages.

---

## 7) Ghi chú cho developers
- Ghi rõ trong README chính: phải cài Poppler trước khi chạy pipeline nếu xử lý PDF.
- Nếu chạy CI (GitHub Actions), thêm step `sudo apt-get install -y poppler-utils` vào workflow.

---

Nếu bạn muốn, tôi sẽ commit các script mẫu vào repo và tạo commit. Bạn có muốn tôi tiếp tục và thử `git push` luôn (nếu repo remote được cấu hình và bạn có quyền) hay chỉ tạo commit và hướng dẫn bạn push từ máy của bạn?