from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
import os
from django.conf import settings
import urllib.parse
from django.shortcuts import render, redirect, get_object_or_404
from .forms import ProfileForm, DocumentForm
from .models import Profile, Document, Interaction
import PyPDF2
import openai
from openai import OpenAI
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
import markdown



def index(request):
    return render(request, 'book_cat_utf8_noruby.html')

def detail(request):
    return HttpResponse("This is the detail page of book_app.")

def upload_profile(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            model = form._meta.model

            return render(request, 'success.html', {'profile': model})
    else:
        form = ProfileForm()
    return render(request, 'upload_profile.html', {'form': form})

def handle_uploaded_file(f):
    # プロジェクトベースディレクトリからの絶対パスを取得
    upload_dir = os.path.join(settings.BASE_DIR, "uploaded_files")
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # ファイルパスを生成
    file_path = os.path.join(upload_dir, f.name)
    
    # ファイルの書き込み
    with open(file_path, "wb") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

def success_view(request):
    return render(request, 'success.html')

def profile_list_view(request):
    profiles = Profile.objects.all()  # すべてのProfileを取得
    return render(request, 'profile_list.html', {'profiles': profiles})

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
            document.text_content = text_content  # 抽出した文字データを保存
            summary = summarize_text(text_content, points = points)
            document.text_summary = summary
            document.save()  # データベースに保存
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

def summarize_text(text, points=None, model="gpt-4o-mini"):
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

@csrf_exempt  # 開発時のみ推奨。本番環境ではCSRFトークンを適切に処理
def chat_with_ai(request, document_id):
    if request.method == 'POST':
        # JSONデータを取得
        data = json.loads(request.body)
        user_message = data.get('message', '')

        # 指定されたドキュメントを取得
        document = Document.objects.get(pk=document_id)

        # ユーザーメッセージを保存
        Interaction.objects.create(document=document, role='user', message=user_message)

        try:
            past_interactions = Interaction.objects.filter(document=document).order_by('timestamp')
            messages = [{"role": "system", "content": "あなたは専門家です。日本語で出力してください。"}]
            for interaction in past_interactions:
                if interaction.role == 'ai':
                    interaction.role = 'assistant'
                messages.append({"role": interaction.role, "content": interaction.message})

            # 新しいユーザーメッセージを追加
            messages.append({"role": "user", "content": user_message})
            
            client = OpenAI()
            model="gpt-4o-mini"
            # ChatGPT APIの呼び出し
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # max_tokens=100,  # 要約結果のトークン数の制限
                temperature=0.5  # 出力の多様性の調整
            )
            # 返答メッセージの抽出
            ai_message = response.choices[0].message.content
            ai_message = markdown.markdown(ai_message)

            # AIのメッセージを保存
            Interaction.objects.create(document=document, role='ai', message=ai_message)


            return JsonResponse({'response': ai_message})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

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
