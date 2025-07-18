
from django.urls import path
from .views import DiseasePredictionAPI,home, disease_prediction, weather_view , register_doctor, doctor_login, doctor_logout, doctor_dashboard , call_doctor   
urlpatterns = [
    path('', home, name='home'),
    path('disease/', disease_prediction, name='disease'),  
    path('predict/', DiseasePredictionAPI.as_view(), name='disease-prediction'),
    path('weather/', weather_view, name='weather'),
    path('doctor/register/', register_doctor, name='doctor-register'),
    path('doctor/login/', doctor_login, name='doctor-login'),   
    path('doctor/logout/', doctor_logout, name='doctor-logout'),
    path('doctor/dashboard/', doctor_dashboard, name='doctor-dashboard'),
    path('doctor/call/', call_doctor, name='call-doctor'),
]