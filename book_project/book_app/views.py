from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
import os
from django.conf import settings
import urllib.parse
from django.shortcuts import render, redirect
from .forms import ProfileForm
from .models import Profile

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


# Create your views here.
