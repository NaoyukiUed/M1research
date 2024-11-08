from openai import OpenAI
from pydantic import BaseModel

class TranslatedText(BaseModel):
    text: str

class TranslatedTexts(BaseModel):
    texts: list[TranslatedText]

def translate_text(texts):
    client = OpenAI()
    messages = [
        {'role': 'system', 'content': '以下の文章をそれぞれ日本語に翻訳してください: '}
    ]

    for text in texts:
        messages.append({'role': 'user', 'content': text})
    response = client.beta.chat.completions.parse(
        model = 'gpt-4o-mini',
        messages = messages,
        response_format=TranslatedTexts
    )
    translated_text = response.choices[0].message.parsed
    return translated_text

hoge = translate_text(["Hello, how are you?,I'm fine"])  # テスト用のテキストを入力
for text in hoge.texts:
    print(text.text)  # 翻訳結果を出力