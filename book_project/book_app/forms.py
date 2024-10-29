from django import forms
from .models import Profile, Document

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['name', 'profile_image']

class DocumentForm(forms.ModelForm):
    point_1 = forms.CharField(max_length=255, required=False, label="知りたいこと 1")
    point_2 = forms.CharField(max_length=255, required=False, label="知りたいこと 2")
    point_3 = forms.CharField(max_length=255, required=False, label="知りたいこと 3")

    class Meta:
        model = Document
        fields = ['title', 'pdf_file', 'point_1', 'point_2', 'point_3']

    def clean(self):
        cleaned_data = super().clean()
        # 質問フィールドをまとめてリストとして保存
        questions = []
        if cleaned_data.get('point_1'):
            questions.append(cleaned_data['point_1'])
        if cleaned_data.get('point_2'):
            questions.append(cleaned_data['point_2'])
        if cleaned_data.get('point_3'):
            questions.append(cleaned_data['point_3'])
        cleaned_data['question_list'] = questions  # リストにまとめた質問をcleaned_dataに追加
        return cleaned_data