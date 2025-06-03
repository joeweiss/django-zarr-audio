from django.urls import path
from .views import audio_proxy_view, health_check

urlpatterns = [
    path("health/", health_check),
    path("proxy/audio/", audio_proxy_view, name="audio-proxy"),
]
