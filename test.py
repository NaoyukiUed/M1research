from pypdf import PdfReader

reader = PdfReader('2312.10997v5.pdf')

page = reader.pages[0]

text = page.extract_text()
print(text)