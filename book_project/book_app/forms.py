from django import forms
from .models import Profile, Document

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['name', 'profile_image']

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('title', 'pdf_file')