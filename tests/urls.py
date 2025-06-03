from django.urls import path, include

urlpatterns = [
    path(
        "zap/",
        include(("django_zarr_audio.urls", "django_zarr_audio"), namespace="zap"),
    ),
]
