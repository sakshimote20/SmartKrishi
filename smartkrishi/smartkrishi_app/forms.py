from django import forms
from django.contrib.auth.models import User
from .models import DoctorProfile

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Enter password'}))

    class Meta:
        model = User
        fields = ['username', 'email', 'password']
        widgets = {
            'username': forms.TextInput(),
            'email': forms.EmailInput(),
        }

class DoctorProfileForm(forms.ModelForm):
    name = forms.CharField(max_length=100, label="Full Name")  # No placeholder
    available_time = forms.CharField(label="Available Time")   # No placeholder

    specialization_input = forms.CharField(label="Specialization (comma-separated)", widget=forms.TextInput(attrs={
        'placeholder': 'e.g. Tomato, Wheat, Cotton'
    }))

    class Meta:
        model = DoctorProfile
        fields = ['name', 'available_time', 'video_link']
        widgets = {
            'video_link': forms.URLInput(),
        }
