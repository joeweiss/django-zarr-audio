from django.urls import path
from .views import (
    add_storage_mapping_view,
    audio_proxy_view,
    browse_storage_view,
    file_info_view,
    health_check,
    list_fsspec_files_view,
    spectrogram_proxy_view,
)

urlpatterns = [
    path("health/", health_check),
    path("proxy/audio/", audio_proxy_view, name="audio-proxy"),
    path("proxy/spectrogram/", spectrogram_proxy_view, name="spectrogram-proxy"),
    path("file-info/", file_info_view, name="file-info"),
    path("list-files/", list_fsspec_files_view, name="list-fsspec-files"),
    path("browse-storage/", browse_storage_view, name="browse-storage"),
    path("browse-storage/<int:mapping_id>/", browse_storage_view, name="browse-storage"),
    path("add-storage-mapping/", add_storage_mapping_view, name="add-storage-mapping"),
]
