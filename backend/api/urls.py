from django.urls import path
from .views import PredictAPIView

urlpatterns = [
    path(r"predict/", PredictAPIView.as_view()),
]
