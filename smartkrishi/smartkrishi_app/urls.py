
from django.urls import path
from .views import DiseasePredictionAPI,home, disease_prediction, weather_view  
urlpatterns = [
    path('', home, name='home'),
    path('disease/', disease_prediction, name='disease'),  
    path('predict/', DiseasePredictionAPI.as_view(), name='disease-prediction'),
    path('weather/', weather_view, name='weather')
]  