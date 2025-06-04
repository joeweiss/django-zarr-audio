from django.urls import path
from .views import audio_proxy_view, health_check, list_fsspec_files_view

urlpatterns = [
    path("health/", health_check),
    path("proxy/audio/", audio_proxy_view, name="audio-proxy"),
    path("list-files/", list_fsspec_files_view, name="list-fsspec-files"),
]
