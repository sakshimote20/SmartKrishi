from django import forms
from .models import WorkPost

class WorkPostForm(forms.ModelForm):
    class Meta:
        model = WorkPost
        fields = ['farmer_name', 'contact_number', 'village', 'crop_type', 'work_description']
