from django.db import models


class Profile(models.Model):
    name = models.CharField(max_length=100)
    profile_image = models.ImageField(upload_to='profile_pics/')  # 画像を保存するパスを指定
# Create your models here.

class Document(models.Model):
    title = models.CharField(max_length=100)
    pdf_file = models.FileField(upload_to='pdfs/')  # アップロード先のディレクトリを指定
    uploaded_at = models.DateTimeField(auto_now_add=True)
    text_content = models.TextField(blank=True)
    text_summary = models.TextField(blank=True)

    def __str__(self):
        return self.title
