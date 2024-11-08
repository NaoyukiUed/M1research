import pdfplumber
import re

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def chunk_text_by_period(text, max_length=3000):
    chunks = []
    current_chunk = ""
    
    for sentence in text.split('. '):  # ピリオドで分割して処理
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ". "  # 句を追加しピリオドを戻す
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def clean_text(text):
    cleaned_text = re.sub(r'\[\d+(\-\d+)?\]', '', text)  # [数字]を削除
    return cleaned_text

def count_periods(text):
    return text.count('. ')

# PDFファイルからテキストを抽出し、チャンキング
pdf_file_path = r'C:\Users\noyku\Desktop\研究\book_project\2312.10997v5.pdf'
text_content = extract_text_from_pdf(pdf_file_path)
cleaned_text = clean_text(text_content)
print(f"Total periods: {count_periods(cleaned_text)}")
chunks = chunk_text_by_period(cleaned_text)
for chunk in chunks:
    print(len(chunk))
print(f"Total chunks: {len(chunks)}")

# 結果を表示
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:\n{chunk}\n")
#     if i ==10:
#         break
