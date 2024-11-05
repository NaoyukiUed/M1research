import PyPDF2
from openai import OpenAI

def extract_text_from_pdf(file_path):
    # PDFファイルを読み込む
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        
        # 全ページのテキストを抽出
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"  # ページごとに改行を追加
        
    return text

# 使用例
pdf_file_path = 'C:/Users/noyku/Desktop/研究/book_project/2312.10997v5.pdf'  # PDFファイルへのパスを指定
text_content = extract_text_from_pdf(pdf_file_path)
# print(text_content)  # 抽出したテキストを表示

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "日本語で出力してください。ユーザの発言にはページ番号や、引用番号、引用先、記号など本文の内容とは関係ない文字や文が多く含まれています。これらの不要なものを取り除いて、本文の内容のみで再度全て出力してください。"},
        {"role": "user", "content": text_content}
    ],
    )

summary = response.choices[0].message.content
print(summary)  # 要約結果を表示
