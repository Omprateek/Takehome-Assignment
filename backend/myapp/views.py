# views.py

from django.shortcuts import render
from .datasets import load_cifar10_dataset, preprocess_cifar10_dataset
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UploadedImage
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

def some_view(request):
    cifar10_data = load_cifar10_dataset('path/to/cifar-10-python.tar.gz')
    X_train, X_test, y_train, y_test = preprocess_cifar10_dataset(cifar10_data)
    return render(request, 'some_template.html', {'X_train': X_train, 'y_train': y_train})

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = UploadedImage(image=request.FILES['image'])
        uploaded_image.save()
        return JsonResponse({'message': 'Image uploaded successfully.'})
    else:
        return JsonResponse({'error': 'Invalid request.'}, status=400)

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_image = UploadedImage(image=request.FILES['image'])
            uploaded_image.save()
            return JsonResponse({'message': 'Image uploaded successfully.'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request.'}, status=400)

@api_view(['POST'])
def upload_image(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        images = request.FILES.getlist('images')
        for image in images:
            with open('path/to/save/' + image.name, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
        # Return a success response
        return Response({'message': 'Images uploaded successfully'}, status=status.HTTP_201_CREATED)
    else:
        # Return an error response if no images were uploaded
        return Response({'error': 'No images uploaded'}, status=status.HTTP_400_BAD_REQUEST)
