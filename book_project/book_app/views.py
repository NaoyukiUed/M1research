from django.urls import reverse
from django.shortcuts import render, redirect, get_object_or_404
from .forms import DocumentForm
from .models import Document, Interaction, SENTENCE
import PyPDF2
from openai import OpenAI
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
import markdown
import numpy as np
from numpy.linalg import norm
from pydantic import BaseModel
import re


class Content(BaseModel):
    id: int
    title: str
    description: str

class StructedToc(BaseModel):
    contents: list[Content]

#資料の読みやすさのために入力された文章の引用番号を削除
#input: str output: str
def clean_text(text):
    cleaned_text = re.sub(r'\[\d+(\-\d+)?\]', '', text)  # [数字]を削除
    return cleaned_text

#ピリオドによるチャンキング
#input: str output: list[str]
def chunk_sentences(text):
    # 正規表現を使って文章を文末の句読点で分割する
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

#文字数によるチャンキング
#input: str output: list[str]
def chunk_text_by_period(text, max_length=300):
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

#PDFファイルをアップロードするページのビュー
def upload_pdf(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)  # まだデータベースに保存しない
            # 知りたいことを取得
            point_1 = form.cleaned_data.get('point_1', '')
            point_2 = form.cleaned_data.get('point_2', '')
            point_3 = form.cleaned_data.get('point_3', '')
            points = [point_1, point_2, point_3]  # 知りたいことをリストにまとめる
            # PDFから文字データを抽出
            pdf_file = request.FILES['pdf_file']
            text_content = extract_text_from_pdf(pdf_file)
            text_content = clean_text(text_content)
            text_content = text_content.replace('\n', '')
            document.text_content = text_content  # 抽出した文字データを保存
            document.question_num = 0
            structed_toc = generate_structed_toc(text_content)
            document.structed_toc = structed_toc.dict()
            document.save()

            #最大文字数を設定し、句点ごとで文章を分割
            chunks = chunk_text_by_period(text_content)

            #chunk化された文章をembedding
            print('len(chunks)', len(chunks))
            for i,chunk in enumerate(chunks):
                if len(chunk) == 0:
                    continue
                print(i,len(chunk))
                embedding = get_embedding(chunk)
                SENTENCE.objects.create(sentence=chunk, document=document, embedding=embedding.tolist())

            print(document)
            
            return redirect('pdf_list')  # アップロード後に一覧ページへリダイレクト
    else:
        form = DocumentForm()
    return render(request, 'upload_pdf.html', {'form': form})

#PDFファイルの一覧を表示するページのビュー
def pdf_list(request):
    documents = Document.objects.all()
    return render(request, 'pdf_list.html', {'documents': documents})

#PDFファイルからテキストを抽出し、文字列にする
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text
    

#入力された文字列に対する埋め込みを作成
def get_embedding(question,model='text-embedding-3-small'):
    client = OpenAI()
    response = client.embeddings.create(input=[question], model=model)
    return np.array(response.data[0].embedding)

#2つのベクトルのコサイン類似度を計算
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

#指定された文書から関連する文章を抽出する
def find_relevant_sentences(document, question):

    #指定された文書に対するchunkのembeddingを取得
    sentence_embeddings = SENTENCE.objects.filter(document=document).values_list('embedding', flat=True)
    # print(SENTENCE.objects.values_list('document', flat=True))
    # print(sentence_embeddings)
    
    #ユーザの入力を翻訳する
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
         messages=[
            {'role': 'system', 'content': '以下のメッセージを英語に翻訳してください。'},
            {'role': 'user', 'content': question},
        ],
        temperature=0.5
    )
    translated_question = response.choices[0].message.content

    # 質問の埋め込みを作成
    question_embedding = get_embedding(translated_question)

    #類似度を計算し、閾値以上の文章を抽出
    threshold = 0.5
    similar_sentences = []
    for i , sentence_embedding in enumerate(sentence_embeddings):
        similarity = cosine_similarity(sentence_embedding, question_embedding)
        if similarity >= threshold:
            similar_sentences.append((i, similarity))
    
    
    return similar_sentences


#StructuredToc型の目次のリストを生成
def generate_structed_toc(text):
    client = OpenAI()

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {'role': 'system', 'content': '日本語でユーザのメッセージの目次とその概要をhtml形式で日本語で作成して'},
            {'role': 'user', 'content': text}
        ],
        response_format=StructedToc
    )
    structed_toc = response.choices[0].message.parsed
    print(structed_toc)
    return structed_toc

#ユーザの出力に対するAIの出力を生成
def generate_response(document, user_message):
    client = OpenAI()
    last_ai_message = Interaction.objects.filter(document=document).latest('timestamp').message
    chunks_related_to_ai_message = find_relevant_sentences(document, last_ai_message)
    chunks_related_to_user_message = find_relevant_sentences(document, user_message)

    related_chunks = []
    for i,element in enumerate(chunks_related_to_ai_message+chunks_related_to_user_message , start= 1):
        related_chunks.append(f"関連する文章{i}:\n{element[0]}")
    related_sentence = "\n".join(related_chunks)
    
    #GPTに対する入力を生成
    chat_messages = document.interactions.all().order_by('timestamp')
    messages = [
        {'role': message.role, 'content': message.message}
        for message in chat_messages
    ]
    messages.insert( 0,{'role': 'system', 'content': f"以下のPDFの内容を基に、敬語を使わないで、人間らしい自然な対話を生成してください。あなたはこの本を深く理解している人として振る舞い、読者と対話します。対話では、読者の意見や感想を引き出し、共感や新しい視点を提供してください。本のテーマ、または著者の意図について議論する具体的な質問を投げかけてください。返答の際には箇条書きは使わないでください。人間らしい自然な対話を意識してください。200文字以内で出力してください。また、話し言葉で出力して。敬語を使わないでください。\n{related_sentence}"})
    messages.append({'role': 'user', 'content': user_message})

    #ユーザの入力に対するAIの出力を生成
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )
    new_ai_message = response.choices[0].message.content
    return new_ai_message

#detailページのビュー
def document_detail(request, pk):
    document = get_object_or_404(Document, pk=pk)
    return render(request, 'document_detail.html', {'document': document})

#ユーザからの入力があったときに呼び出され、AIからの出力を返す
@csrf_exempt
def chat_with_ai(request, document_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')
        document = Document.objects.get(pk=document_id)
        Interaction.objects.create(document=document, role='user', message=user_message)

        #ChatGPTのAPIから出力を得る
        response = generate_response(document, user_message)
        Interaction.objects.create(document=document, role='assistant', message=response)

        return JsonResponse({'response': response})
        
        

# 過去のやり取りを取得する
def get_chat_history(request, document_id):
    document = Document.objects.get(pk=document_id)
    chat_messages = document.interactions.all().order_by('timestamp')
    chat_data = [
        {'role': message.role, 'message': message.message, 'timestamp': message.timestamp}
        for message in chat_messages
    ]
    return JsonResponse({'chat_history': chat_data})

#不要になった資料を削除する
def document_delete(request, pk):
    document = get_object_or_404(Document, pk=pk)
    if request.method == "POST":
        document.delete()
        return redirect('pdf_list')  # 削除後、リストページにリダイレクト
    return redirect(reverse('document_detail', args=[pk]))
