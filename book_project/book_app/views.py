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

class Evaluation(BaseModel):
    comprehension_level: int
    not_understand: list[str]

class ExistQuestion(BaseModel):
    exist: bool

class Content(BaseModel):
    id: int
    title: str
    description: str

class StructedToc(BaseModel):
    contents: list[Content]

class Question(BaseModel):
    child_num: list[int]
    content_id: int
    question: str
    question_embeddingf: list[float]
    answer: str
    answer_embedding: list[float]

class QuestionList(BaseModel):
    questions: list[Question]

class ConversationContent(BaseModel):
    review: str
    questions: QuestionList

class TranslatedText(BaseModel):
    text: str

class TranslatedTexts(BaseModel):
    texts: list[TranslatedText]

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

            chunks = chunk_text_by_period(text_content)
            chunks = translate_text(chunks)
            
            # for i,chunk in enumerate(chunks.texts):
            #     print(i)
            #     embedding = get_embedding(chunk.text)
            #     SENTENCE.objects.create(sentence=chunk.text, document=document, embedding=embedding.tolist())

            

            # questions = generate_structed_questions(structed_toc, document)
            # document.question_list = questions.dict()
            # document.save()
            # for child in reversed(questions.dict()['questions']):
            #     print(child['question'])
            #     question_embedding = get_embedding(child['question'])
            #     answer_embedding = get_embedding(child['answer'])

                

            #     similar_answers = find_relevant_sentences(document, child['answer'])
            #     if len(similar_answers) == 0:
            #         continue
            #     child['question_embedding'] = question_embedding.tolist()
            #     child['answer_embedding'] = answer_embedding.tolist()
            #     document.question_stack.append(child)
            #     document.save()
            
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

#入力された文字列のリストに対する埋め込みのリストを作成
def get_embeddings(texts, model='text-embedding-3-small'):
    client = OpenAI()
    embeddings = []
    chunks = chunk_sentences(texts)
    for chunk in chunks:
        response = client.embeddings.create(input = [chunk], model = model).data[0].embedding
        embeddings.append(response)
    return np.array(embeddings)
    

#入力された文字列に対する埋め込みを作成
def get_embedding(question,model='text-embedding-3-small'):
    client = OpenAI()
    response = client.embeddings.create(input=[question], model=model)
    return np.array(response.data[0].embedding)

#2つのベクトルのコサイン類似度を計算
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

#入力された文章に対する類似度が高いチャンクを抽出する
def find_similar_sentences(sentence_embeddings, question_embedding, threshold=0.6):
    similar_sentences = []
    for i, sentence_embedding in enumerate(sentence_embeddings):
        similarity = cosine_similarity(sentence_embedding, question_embedding)
        if similarity >= threshold:
            similar_sentences.append((i, similarity))  # ページ番号は1から始まる
    return similar_sentences

#指定された文書から関連する文章を抽出する
def find_relevant_sentences(document, question):

    sentence_embeddings = SENTENCE.objects.filter(document=document).values_list('embedding', flat=True)
    
    # 3. 質問の埋め込みを作成
    question_embedding = get_embedding(question)
    
    # 4. 類似度の計算
    similar_sentences = find_similar_sentences(sentence_embeddings, question_embedding)
    
    return similar_sentences

#入力された文章を翻訳する
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

#QuestionList型の質問リストを生成
def generate_structed_questions(structed_toc, document):
    questions = []
    client = OpenAI()
    messages = [
            {'role': 'system', 'content': f"日本語でユーザのメッセージは目次とその概要です。目次の各テーマに対する質問とその回答を5個ずつ作成して。ただし、作成する質問に対する答えが以下の文章に含まれるようにして。また、parent_numは-1にして\n {document.text_content}"}
        ]
    for content in structed_toc.contents:
        messages.append({'role': 'user', 'content': f"id:{content.id} title:{content.title} description{content.description}\n"})
    print(messages)
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        response_format=QuestionList
    )
    questions = response.choices[0].message.parsed
    return questions


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

#文字列型の目次を作成
def generate_toc(text):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {'role': 'system', 'content': '日本語でユーザのメッセージの目次とその概要をhtml形式で日本語で作成して'},
            {'role': 'user', 'content': text}
        ],
        temperature=0.5
    )
    toc= response.choices[0].message.content
    return toc

#人間の出力に対するAIの出力を生成
def generate_conversation_content(document, user_message):
    client = OpenAI()

    print(user_message)

    
    chat_messages = document.interactions.all().order_by('timestamp')
    messages = [
        {'role': message.role, 'content': message.message}
        for message in chat_messages
    ]
    print(messages)
    messages.insert( 0,{'role': 'system', 'content': f"以下のPDFの内容を基に、敬語を使わないで、人間らしい自然な対話を生成してください。あなたはこの本を深く理解している人として振る舞い、読者と対話します。対話では、読者の意見や感想を引き出し、共感や新しい視点を提供してください。本のテーマ、または著者の意図について議論する具体的な質問を投げかけてください。返答の際には箇条書きは使わないでください。人間らしい自然な対話を意識してください。200文字以内で出力してください。また、話し言葉で出力して。敬語を使わないでください。\n{document.text_content}"})
    
    messages.append({'role': 'user', 'content': user_message})
    # print(messages)

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )

    conversation_content = response.choices[0].message.content

    # question = document.question_stack[-1]
    # response = client.beta.chat.completions.parse(
    #     model="gpt-4o-mini-2024-07-18",
    #     messages=[
    #         {'role': 'system', 'content': f"以下のPDFの内容を基に、人間らしい自然な対話を生成してください。あなたはこの本を深く理解している人として振る舞い、読者と対話します。対話では、読者の意見や感想を引き出し、共感や新しい視点を提供してください。本のテーマ、または著者の意図について議論する具体的な質問を投げかけてください。\n{document.text_content}"},
    #         {'role': 'user', 'content': f"質問:{question['question']}\nAIの回答:{question['answer']}\nユーザの回答:{user_message}"}
    #     ],
    #     response_format=ConversationContent
    # )
    # conversation_content = response.choices[0].message.parsed
    # document.save()
    return conversation_content

#入力された文書を要約する
def summarize_text(text, points=None, model="gpt-4o"):
    client = OpenAI()

    if points:
        points_text = "\n".join(f"- {point}" for point in points if point)
    else:
        points_text = "特に知りたいことは指定されていません。"

    # ChatGPT APIの呼び出し
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは専門家です。日本語で出力してください。"},
            {"role": "user", "content": f"以下の内容を要約してください:\n\n{text}。また、以下の知りたいことを考慮に入れてください:\n{points_text}"}
        ],
        # max_tokens=100,  # 要約結果のトークン数の制限
        temperature=0.5  # 出力の多様性の調整
    )
    # 返答メッセージの抽出
    summary = response.choices[0].message.content
    summary = markdown.markdown(summary)
    return summary


#detailページのビュー
def document_detail(request, pk):
    document = get_object_or_404(Document, pk=pk)
    return render(request, 'document_detail.html', {'document': document})

#ユーザの発言に質問が含まれているかのExistQuesion型を出力
def analyze_comment(comment):
    client = OpenAI
    completion =client.beta.chat.completions.parse(
        model='gpt-4o-mini-2024-07-18',
        messages=[
            {'role': 'system', 'content': 'ユーザの発言に質問が含まれているか教えて'},
            {'role': 'user', 'content': comment}
        ],
        response_format=ExistQuestion
    )

#ユーザからの入力があったときに呼び出され、AIからの出力を返す
@csrf_exempt
def chat_with_ai(request, document_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')

        document = Document.objects.get(pk=document_id)
        
        print(document.question_progress)

        Interaction.objects.create(document=document, role='user', message=user_message)

        #最初のやり取りを決定する部分
        # if document.question_progress == 0:
        #     if len(document.question_stack) == 0:
        #         ai_message = f"質問のスタックが空です。"
        #         return JsonResponse({'response': ai_message})
        #     ai_message = document.question_stack[-1]['question']
        #     Interaction.objects.create(document=document, role='ai', message=ai_message)
        #     document.question_progress = 1
        #     document.save()
        #     return JsonResponse({'response': ai_message})
        
        #ChatGPTのAPIから出力を得る
        conversation_content = generate_conversation_content(document, user_message)
        print(conversation_content)
        

        #生成された質問をembeddingして、似た質問がないか確認し、データベースに格納
        # questions = conversation_content.questions
        # for child in reversed(questions.dict()['questions']):
        #     question_embedding = get_embedding(child['question'])
        #     answer_embedding = get_embedding(child['answer'])

            

        #     similar_answers = find_relevant_sentences(document, child['answer'])
        #     if len(similar_answers) == 0:
        #         continue
           

        #     child['question_embedding'] = question_embedding.tolist()
        #     child['answer_embedding'] = answer_embedding.tolist()
        #     document.question_stack.append(child)
        #     document.save()

        #関連度の高い文章を出力
        # if len(document.question_stack) == 0:
        #     return JsonResponse({'response': ai_message})
        # new_question = document.question_stack[-1]
        # print(new_question)
        # relevant_sentences = find_relevant_sentences(document, new_question['answer'])
        # print(relevant_sentences)
        # relevant_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)
        # print(relevant_sentences)
        # sentences = SENTENCE.objects.filter(document=document).values_list('sentence', flat=True)
        
        #AIからの返答を出力

        # question = document.question_stack.pop()
        ai_message =conversation_content
        # document.save()

        # if len(document.question_stack) == 0:
        #     return JsonResponse({'response': ai_message})
        # ai_message = f"{ai_message}<h2>次の質問</h2><div>{new_question['question']}</div>"

        # for i,sentence in enumerate(relevant_sentences):
        #     ai_message = f"{ai_message}<h2>関連する文章{i+1}</h2><div>{sentences[sentence[0]]}</div>"

        
        Interaction.objects.create(document=document, role='assistant', message=ai_message)

        # new_question_list = conversation_content.questions.dict()
        # document.question_list['questions'].extend(new_question_list['questions'])
        document.save()
        return JsonResponse({'response': ai_message})
        
        

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


# Create your views here.
