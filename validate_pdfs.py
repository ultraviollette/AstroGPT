import os
import glob
from PyPDF2 import PdfReader

PDF_DIR = "./university_docs/"
QUARANTINE_DIR = "./quarantine/"

# 격리 폴더 없으면 생성
os.makedirs(QUARANTINE_DIR, exist_ok=True)

def is_valid_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            PdfReader(f)
        return True
    except Exception as e:
        return False

def main():
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    total = len(pdf_files)
    print(f"🔍 Scanning {total} PDF files...\n")

    valid, invalid = 0, 0
    for path in pdf_files:
        filename = os.path.basename(path)
        if is_valid_pdf(path):
            print(f"✅ OK: {filename}")
            valid += 1
        else:
            print(f"❌ Corrupt: {filename} → moved to quarantine")
            os.rename(path, os.path.join(QUARANTINE_DIR, filename))
            invalid += 1

    print("\n🧾 Summary:")
    print(f"✔️ Valid PDFs: {valid}")
    print(f"❌ Invalid PDFs: {invalid}")
    print(f"📁 Quarantine folder: {QUARANTINE_DIR}")

if __name__ == "__main__":
    main()