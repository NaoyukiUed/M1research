from django.urls import path
from . import views  # views.pyをインポート
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('pdfs/', views.pdf_list, name='pdf_list'),
    path('document/<int:pk>/', views.document_detail, name='document_detail'),
    path('chat-with-ai/<int:document_id>/', views.chat_with_ai, name='chat_with_ai'),
    path('get-chat-history/<int:document_id>/', views.get_chat_history, name='get_chat_history'),
    path('document/<int:pk>/delete/', views.document_delete, name='document_delete'),
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])