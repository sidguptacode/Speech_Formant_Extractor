from django.urls import path

from . import views

urlpatterns = [
    # ex: /speech_analyzer/
    path('', views.index, name='index'),
]
