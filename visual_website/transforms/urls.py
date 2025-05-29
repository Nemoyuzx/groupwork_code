from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('fourier/', views.fourier_transform, name='fourier'),
    path('laplace/', views.laplace_transform, name='laplace'),
    path('wavelet/', views.wavelet_transform, name='wavelet'),
    path('hough/', views.hough_transform, name='hough'),
    path('z_transform/', views.z_transform, name='z_transform'),
    path('fractional_fourier/', views.fractional_fourier_transform, name='fractional_fourier'),
    path('api/transform/', views.api_transform, name='api_transform'),
]
