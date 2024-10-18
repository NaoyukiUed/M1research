from django.urls import path
from . import views  # views.pyをインポート
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),  # 例: /book_app/
    path('detail/', views.detail, name='detail'),  # 例: /book_app/detail/
    path('form/', views.upload_profile, name='form'),
    path('success/url/', views.success_view, name='upload_success'),
    path('profiles/', views.profile_list_view, name='profile_list'),  # 一覧表示
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0]) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)