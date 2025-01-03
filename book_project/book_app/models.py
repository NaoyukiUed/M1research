from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=100)
    pdf_file = models.FileField(upload_to='pdfs/')  # アップロード先のディレクトリを指定
    uploaded_at = models.DateTimeField(auto_now_add=True)
    text_content = models.TextField(blank=True)
    structed_toc = models.JSONField(blank=True, default=list)

    def __str__(self):
        return self.title
    
class Interaction(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('ai', 'AI'),
    ]

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='interactions')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_role_display()}: {self.message[:50]}..."  # メッセージの最初の50文字を表示
    
class SENTENCE(models.Model):
    sentence = models.TextField()
    embedding = models.JSONField(default=list, blank=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='sentences')
    def __str__(self):
        return self.sentence
