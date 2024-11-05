import pdfplumber
from transformers import AutoTokenizer

# トークナイザーの初期化（GPT-3/4用に調整）
# tokenizer = AutoTokenizer.from_pretrained("gpt-2")

def extract_paragraphs_from_pdf(pdf_path):
    paragraphs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # 改行で段落を分割
                page_paragraphs = text.split('\n\n')
                paragraphs.extend([para.strip() for para in page_paragraphs if para.strip()])
    return paragraphs

# def chunk_paragraphs(paragraphs, max_tokens=512):
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for para in paragraphs:
#         token_length = len(tokenizer.tokenize(para))
        
#         if current_length + token_length <= max_tokens:
#             current_chunk.append(para)
#             current_length += token_length
#         else:
#             # チャンクを保存し、次のチャンクを開始
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [para]
#             current_length = token_length
    
#     # 最後のチャンクを保存
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
    
#     return chunks

# PDFから段落を抽出
pdf_path = "C:\\Users\\noyku\\Desktop\\研究\\book_project\\2312.10997v5.pdf"
paragraphs = extract_paragraphs_from_pdf(pdf_path)
for i in range(10):
    print(i)
    print(paragraphs[i])

# # 段落をチャンキング
# chunks = chunk_paragraphs(paragraphs, max_tokens=512)

# # チャンクの内容を表示
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:")
#     print(chunk)
#     print("\n---\n")