
from django.urls import path
from .views import DiseasePredictionAPI,home, disease_prediction
urlpatterns = [
    path('', home, name='home'),
    path('disease/', disease_prediction, name='disease'),  
    path('predict/', DiseasePredictionAPI.as_view(), name='disease-prediction'),
]  