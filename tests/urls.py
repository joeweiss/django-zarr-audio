from django.urls import path
from django_zarr_audio.views import health_check

urlpatterns = [
    path("health/", health_check),
]
