# admin.py

from django.contrib import admin
from .models import StorageAccessProfile, StorageMapping, AudioFile


@admin.register(StorageAccessProfile)
class StorageAccessProfileAdmin(admin.ModelAdmin):
    list_display = (
        "credentials_label",
        "backend",
        "status",
        "description",
        "created",
        "modified",
    )
    list_filter = ("backend", "status")
    search_fields = ("credentials_label", "description")
    ordering = ("credentials_label",)


@admin.register(StorageMapping)
class StorageMappingAdmin(admin.ModelAdmin):
    list_display = (
        "input_prefix",
        "input_profile",
        "output_profile",
        "output_base_uri",
        "status",
        "created",
        "modified",
    )
    list_filter = ("status",)
    search_fields = ("input_prefix", "output_base_uri")
    raw_id_fields = ("input_profile", "output_profile")
    ordering = ("input_prefix",)


@admin.register(AudioFile)
class AudioFileAdmin(admin.ModelAdmin):
    list_display = (
        "uri",
        "status",
        "storage_mapping",
        "zarr_uri",
        "created",
        "modified",
    )
    list_filter = ("status",)
    search_fields = ("uri", "zarr_uri")
    raw_id_fields = ("storage_mapping",)
    readonly_fields = ("created", "modified")
    ordering = ("-created",)
