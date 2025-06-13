from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from PIL import Image
from ml_models.predictor import predict_image  

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
