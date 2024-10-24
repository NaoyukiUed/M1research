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