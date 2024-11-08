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

def clean_text(text):
    cleaned_text = re.sub(r'\[\d+(\-\d+)?\]', '', text)  # [数字]を削除
    return cleaned_text

#ピリオドによるチャンキング
def chunk_sentences(text):
    # 正規表現を使って文章を文末の句読点で分割する
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

#文字数によるチャンキング
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
            # summary = summarize_text(text_content, points = points)
            # document.text_summary = summary
            # document.question_list = form.cleaned_data['question_list']
            document.question_num = 0
            structed_toc = generate_structed_toc(text_content)
            document.structed_toc = structed_toc.dict()
            document.save()

            chunks = chunk_text_by_period(text_content)
            chunks = translate_text(chunks)
            
            for i,chunk in enumerate(chunks.texts):
                print(i)
                embedding = get_embedding(chunk.text)
                SENTENCE.objects.create(sentence=chunk.text, document=document, embedding=embedding.tolist())

            

            questions = generate_structed_questions(structed_toc, document)
            document.question_list = questions.dict()
            document.save()
            for child in reversed(questions.dict()['questions']):
                print(child['question'])
                question_embedding = get_embedding(child['question'])
                answer_embedding = get_embedding(child['answer'])

                # similar_sentences = find_similar_sentences(document.question_stack, question_embedding)
                # similar_sentences = sorted(similar_sentences, key=lambda x: x[1], reverse=True)
                # if len(similar_sentences) == 0:
                #     continue
                # best_similarity = similar_sentences[0][1]
                # if best_similarity > 0.8:
                #     continue

                similar_answers = find_relevant_sentences(document, child['answer'])
                # similar_answers = sorted(similar_answers, key=lambda x: x[1], reverse=True)
                if len(similar_answers) == 0:
                    continue
                # best_similarity = similar_answers[0][1]
                # if best_similarity < 0.6:
                #     continue
                child['question_embedding'] = question_embedding.tolist()
                child['answer_embedding'] = answer_embedding.tolist()
                document.question_stack.append(child)
                document.save()

                

            # toc = ''
            # for content in structed_toc.contents:
            #     toc = toc + f'<br><h2>{content.title}</h2>'
            #     toc = toc + f'<br><p>{content.description}</p>'
            # document.toc = toc

            # document.save()  # データベースに保存


            

            
            return redirect('pdf_list')  # アップロード後に一覧ページへリダイレクト
    else:
        form = DocumentForm()
    return render(request, 'upload_pdf.html', {'form': form})

def pdf_list(request):
    documents = Document.objects.all()
    return render(request, 'pdf_list.html', {'documents': documents})

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_embeddings(texts, model='text-embedding-3-small'):
    client = OpenAI()
    embeddings = []
    chunks = chunk_sentences(texts)
    for chunk in chunks:
        response = client.embeddings.create(input = [chunk], model = model).data[0].embedding
        embeddings.append(response)
    return np.array(embeddings)
    # for text in texts:
    #     text = text.replace('\n','')
    #     response = client.embeddings.create(input = [text], model = model).data[0].embedding
    #     embeddings.append(response)
    # return np.array(embeddings)

def get_embedding(question,model='text-embedding-3-small'):
    client = OpenAI()
    response = client.embeddings.create(input=[question], model=model)
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def find_similar_sentences(sentence_embeddings, question_embedding, threshold=0.5):
    similar_sentences = []
    for i, sentence_embedding in enumerate(sentence_embeddings):
        similarity = cosine_similarity(sentence_embedding, question_embedding)
        if similarity >= threshold:
            similar_sentences.append((i + 1, similarity))  # ページ番号は1から始まる
    return similar_sentences

def find_relevant_sentences(document, question):

    sentence_embeddings = SENTENCE.objects.filter(document=document).values_list('embedding', flat=True)
    
    # 3. 質問の埋め込みを作成
    question_embedding = get_embedding(question)
    
    # 4. 類似度の計算
    similar_sentences = find_similar_sentences(sentence_embeddings, question_embedding)
    
    return similar_sentences

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

def generate_conversation_content(document, user_message):
    client = OpenAI()

    question = document.question_stack[-1]

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {'role': 'system', 'content': f"日本語でユーザの入力内容を以下のAIの回答と比較して評価して。また、元の質問内容をさらに深堀するpdf中に答えが存在する質問とその質問に対する回答を3つ作成してください。また、回答を生成する際は以下のpdfの情報を参考にしてください。\n{document.text_content}"},
            {'role': 'user', 'content': f"質問:{question['question']}\nAIの回答:{question['answer']}\nユーザの回答:{user_message}"}
        ],
        response_format=ConversationContent
    )
    conversation_content = response.choices[0].message.parsed
    document.save()
    return conversation_content


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


def document_detail(request, pk):
    document = get_object_or_404(Document, pk=pk)
    return render(request, 'document_detail.html', {'document': document})

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

@csrf_exempt
def chat_with_ai(request, document_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')

        document = Document.objects.get(pk=document_id)
        
        print(document.question_progress)

        Interaction.objects.create(document=document, role='user', message=user_message)

        if document.question_progress == 0:
            if len(document.question_stack) == 0:
                ai_message = f"質問のスタックが空です。"
                return JsonResponse({'response': ai_message})
            ai_message = document.question_stack[-1]['question']
            Interaction.objects.create(document=document, role='ai', message=ai_message)
            document.question_progress = 1
            document.save()
            return JsonResponse({'response': ai_message})
        
        conversation_content = generate_conversation_content(document, user_message)
        print(conversation_content)
        question = document.question_stack.pop()
        ai_message = f"<h2>質問</h2><div>{question['question']}</div><h2>AIの回答</h2><div>{question['answer']}</div><h2>ユーザの回答の評価</h2><div>{conversation_content.review}</div>"
        document.save()

        if len(document.question_stack) == 0:
            return JsonResponse({'response': ai_message})

        questions = conversation_content.questions
        # document.question_list = questions.dict()
        for child in reversed(questions.dict()['questions']):
            question_embedding = get_embedding(child['question'])
            answer_embedding = get_embedding(child['answer'])

            # similar_sentences = find_similar_sentences(document.question_stack, question_embedding)
            # similar_sentences = sorted(similar_sentences, key=lambda x: x[1], reverse=True)
            # print(similar_sentences)
            # if len(similar_sentences) == 0:
            #     continue
            # best_similarity = similar_sentences[0][1]
            # if best_similarity > 0.8:
            #     continue

            similar_answers = find_relevant_sentences(document, child['answer'])
            # similar_answers = sorted(similar_answers, key=lambda x: x[1], reverse=True)
            if len(similar_answers) == 0:
                continue
            # best_similarity = similar_answers[0][1]
            # if best_similarity < 0.6:
            #     continue

            child['question_embedding'] = question_embedding.tolist()
            child['answer_embedding'] = answer_embedding.tolist()
            document.question_stack.append(child)
            document.save()

        if len(document.question_stack) == 0:
            return JsonResponse({'response': ai_message})
        new_question = document.question_stack[-1]
        relevant_sentences = find_relevant_sentences(document, new_question['answer'])
        relevant_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)
        sentences = SENTENCE.objects.filter(document=document).values_list('sentence', flat=True)
        sentence = relevant_sentences[0][0]
        
        ai_message = f"{ai_message}<h2>次の質問</h2><div>{new_question['question']}</div><h2>関連する文章</h2><div>{sentences[sentence]}</div>"

        
        Interaction.objects.create(document=document, role='ai', message=ai_message)

        new_question_list = conversation_content.questions.dict()
        document.question_list['questions'].extend(new_question_list['questions'])
        document.save()
        return JsonResponse({'response': ai_message})
        
        


# @csrf_exempt  # 開発時のみ推奨。本番環境ではCSRFトークンを適切に処理
# def chat_with_ai(request, document_id):
#     if request.method == 'POST':
#         # JSONデータを取得
#         data = json.loads(request.body)
#         user_message = data.get('message', '')

#         # 指定されたドキュメントを取得
#         document = Document.objects.get(pk=document_id)

#         # ユーザーメッセージを保存
#         Interaction.objects.create(document=document, role='user', message=user_message)

#         if document.question_progress == 0:
#             print(document.question_list['questions'])
#             question = document.question_stack.pop()
#             ai_message = question['question']
#             Interaction.objects.create(document=document, role='ai', message=ai_message)
#             document.question_progress = 1
#             document.save()
#             return JsonResponse({'response': ai_message})

#         try:
#             if document.question_num > 8:
#                 # document.question_progress = 0
#                 # document.question_num = 0
#                 # document.save()
#                 # return JsonResponse({'response': '回答の評価と質問の深堀り'})


#                 # past_interactions = Interaction.objects.filter(document=document).order_by('timestamp')
#                 # messages = [{"role": "system", "content": f"あなたは専門家です。日本語で出力してください。ただし、回答の際には以下の情報を参考にしてください。{document.text_content}"}]
#                 # #past_interactionsの中で最も新しいaiのメッセージを取得
#                 # if len(past_interactions) > 0:
#                 #     ai_last_message = past_interactions.filter(role='ai').last().message

#                 # messages.append({"role": "assistant", "content": ai_last_message})
#                 # messages.append({"role": "user", "content": f"質問に対する以下のユーザの回答を深ぼる質問をhtml形式で一つ作成して{user_message}"})

                
#                 # client = OpenAI()
#                 # model="gpt-4o-mini"
#                 # # ChatGPT APIの呼び出し
#                 # response = client.chat.completions.create(
#                 #     model=model,
#                 #     messages=messages,
#                 #     # max_tokens=100,  # 要約結果のトークン数の制限
#                 #     temperature=0.5,  # 出力の多様性の調整
#                 #     # response_format=Evaluation,
#                 # )
#                 # # 返答メッセージの抽出
#                 # ai_message = response.choices[0].message.content
#                 # relevant_sentences = find_relevant_sentences(ai_message)
#                 # #関連度順にソート
#                 # relevant_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)

#                 # sentences = SENTENCE.objects.all().values_list('sentence', flat=True)

#                 # #最も関連度の高いページを取得
#                 # sentence = relevant_sentences[0][0]
#                 # ai_message = f"{ai_message}\n\n関連する文章: {sentences[sentence]}"

#                 # # AIのメッセージを保存
#                 # Interaction.objects.create(document=document, role='ai', message=ai_message)

#                 conversation_content = generate_conversation_content(document, user_message)
#                 question = document.question_stack.pop()
#                 ai_message = f"質問:{question['question']}\nAIの回答:{question['answer']}\n"
#                 ai_message = ai_message + conversation_content.review

#                 new_question_list = conversation_content.questions.dict()
#                 document.question_list['questions'].extend(new_question_list['questions'])
#                 # document.save()

#                 Interaction.objects.create(document=document, role='ai', message=ai_message)


#                 return JsonResponse({'response': ai_message})
#             else:
#                 past_interactions = Interaction.objects.filter(document=document).order_by('timestamp')
#                 messages = [{"role": "system", "content": f"あなたは専門家です。日本語で出力してください。ただし、回答の際には以下の情報を参考にしてください。{document.text_content}"}]
#                 #past_interactionsの中で最も新しいaiのメッセージを取得
#                 if len(past_interactions) > 0:
#                     ai_last_message = past_interactions.filter(role='ai').last().message

#                 messages.append({"role": "assistant", "content": ai_last_message})


#                 # for interaction in past_interactions:
#                 #     if interaction.role == 'ai':
#                 #         interaction.role = 'assistant'
#                 #     messages.append({"role": interaction.role, "content": interaction.message})

#                 shortage_question = f'まず最初に質問に対するAIの回答をpdfの内容を基に簡潔に出力してください。それから、以下の内容が正しい内容を述べているか作成した回答と比較して評価してください。また、不足している部分や補足すべきな点があればその内容に関して問うpdf中に答えが存在する質問を一つ作成してください。{user_message}'

#                 relevant_question = f'まず最初に質問に対するAIの回答をpdfの内容を基に簡潔に出力してください。この時、ユーザの発言は無視してください。それから、以下の内容が正しい内容を述べているか作成した回答と比較して評価してください。また、質問内容、回答内容、pdfの内容に関連のあるpdf中に答えが存在する質問を一つ作成してください。{user_message}'

#                 # 新しいユーザーメッセージを追加
#                 messages.append({"role": "user", "content": shortage_question})
                
#                 client = OpenAI()
#                 model="gpt-4o-mini"
#                 # ChatGPT APIの呼び出し
#                 response = client.chat.completions.create(
#                     model=model,
#                     messages=messages,
#                     # max_tokens=100,  # 要約結果のトークン数の制限
#                     temperature=0.5,  # 出力の多様性の調整
#                     # response_format=Evaluation,
#                 )
#                 # 返答メッセージの抽出
#                 ai_message = response.choices[0].message.content
#                 ai_message = markdown.markdown(ai_message)
#                 # comprehension_level = ai_message.comprehension_level
#                 # not_understand = ai_message.not_understand
#                 # ai_message = f"comprehension_level:{comprehension_level}\n not_understand:{not_understand}"

#                 relevant_sentences = find_relevant_sentences(ai_message)
#                 #関連度順にソート
#                 relevant_sentences = sorted(relevant_sentences, key=lambda x: x[1], reverse=True)

#                 sentences = SENTENCE.objects.all().values_list('sentence', flat=True)

#                 #最も関連度の高いページを取得
#                 sentence = relevant_sentences[0][0]
#                 ai_message = f"{ai_message}\n\n関連する文章: {sentences[sentence]}"

#                 # AIのメッセージを保存
#                 Interaction.objects.create(document=document, role='ai', message=ai_message)
#                 # document.question_progress = 0
#                 document.question_num = 9
#                 document.save() 


#                 return JsonResponse({'response': ai_message})
            
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

# 過去のやり取りを取得するビュー
def get_chat_history(request, document_id):
    document = Document.objects.get(pk=document_id)
    chat_messages = document.interactions.all().order_by('timestamp')
    chat_data = [
        {'role': message.role, 'message': message.message, 'timestamp': message.timestamp}
        for message in chat_messages
    ]
    return JsonResponse({'chat_history': chat_data})

def document_delete(request, pk):
    document = get_object_or_404(Document, pk=pk)
    if request.method == "POST":
        document.delete()
        return redirect('pdf_list')  # 削除後、リストページにリダイレクト
    return redirect(reverse('document_detail', args=[pk]))


# Create your views here.
