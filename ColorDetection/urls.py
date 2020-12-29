from django.urls import path
from . import views

urlpatterns = [
    path('detect', views.detect),
    path('train', views.train),
    path('insert_data', views.insert_data),
    path('get_list_color', views.get_list_color)
]
