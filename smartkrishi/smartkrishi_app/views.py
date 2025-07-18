from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from PIL import Image
from ml_models.predictor import predict_image 
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, DoctorProfileForm
from .models import DoctorProfile,Crop
from django.shortcuts import redirect, render
from django.contrib import messages



class DiseasePredictionAPI(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        try:
            image_file = request.FILES.get("image")
            if not image_file:
                return Response({"error": "Image not found in request."}, status=status.HTTP_400_BAD_REQUEST)

            image = Image.open(image_file)
            result = predict_image(image)

            return Response(result, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

def home(request):
    return render(request, 'smartkrishi_app/home.html')

def disease_prediction(request):
    return render(request, 'smartkrishi_app/disease.html')

def weather_view(request):
    return render(request, 'smartkrishi_app/weather.html')

def call_doctor(request):
    crop_name = request.GET.get('crop', '').strip()
    doctors = []
    message = ""

    # If crop name is entered
    if crop_name:
        try:
            crop = Crop.objects.get(name__iexact=crop_name)
            doctors = DoctorProfile.objects.filter(specialization=crop, is_active=True)

            if not doctors:
                message = f" Currently no doctor available for '{crop_name}'."
        except Crop.DoesNotExist:
            message = f" No specialization found for crop '{crop_name}'."

    # If no crop name entered and no doctors in system at all
    elif not DoctorProfile.objects.exists():
        message = " No crop doctors are currently registered. Please try again later."

    return render(request, 'smartkrishi_app/call_doctor.html', {
        'doctors': doctors,
        'message': message,
        'searched_crop': crop_name
    })
def register_doctor(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        profile_form = DoctorProfileForm(request.POST)

        if user_form.is_valid() and profile_form.is_valid():
            user = user_form.save(commit=False)
            user.set_password(user_form.cleaned_data['password'])
            user.save()

            profile = profile_form.save(commit=False)
            profile.user = user
            profile.name = profile_form.cleaned_data['name']
            profile.save()

            crop_input = profile_form.cleaned_data['specialization_input']
            crop_names = [c.strip().capitalize() for c in crop_input.split(',') if c.strip()]

            for crop_name in crop_names:
                crop, _ = Crop.objects.get_or_create(name=crop_name)
                profile.specialization.add(crop)

            return redirect('doctor-login')
    else:
        user_form = UserRegistrationForm()
        profile_form = DoctorProfileForm()

    return render(request, 'smartkrishi_app/doctor_register.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })

def doctor_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('doctor-dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'smartkrishi_app/doctor_login.html', {'form': form})

def doctor_logout(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('home')

@login_required
def doctor_dashboard(request):
    profile = DoctorProfile.objects.get(user=request.user)
    if request.method == 'POST':
        form = DoctorProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
    else:
        form = DoctorProfileForm(instance=profile)
    return render(request, 'smartkrishi_app/doctor_dashboard.html', {'form': form})
