from django.db import models


class Profile(models.Model):
    name = models.CharField(max_length=100)
    profile_image = models.ImageField(upload_to='profile_pics/')  # 画像を保存するパスを指定
# Create your models here.
