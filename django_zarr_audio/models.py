from django.db import models
from model_utils.models import TimeStampedModel, StatusModel
from model_utils import Choices


class StorageAccessProfile(TimeStampedModel, StatusModel):
    """
    Represents a credentialed access profile for a specific storage backend.
    Used to resolve fsspec-compatible access to input or output storage.
    """

    STATUS = Choices("active", "inactive")

    credentials_label = models.CharField(
        max_length=128,
        db_index=True,
        help_text="Key used to look up credentials for this storage backend (e.g., passed to get_fs_from_env())",
    )

    backend = models.CharField(
        max_length=50,
        choices=[
            ("s3", "S3"),
            ("gcs", "GCS"),
            ("http", "HTTP"),
            ("file", "Local FS"),
            ("custom", "Custom"),
        ],
        help_text="fsspec-compatible backend name",
    )

    description = models.TextField(blank=True)

    def __str__(self):
        return f"{self.credentials_label} ({self.backend})"


class StorageMapping(TimeStampedModel, StatusModel):
    """
    Maps an input prefix to a pair of storage contexts (read/write) and defines
    the base output URI where Zarr-encoded results will be written.
    """

    STATUS = Choices("active", "inactive")

    input_prefix = models.CharField(
        max_length=255,
        unique=True,
        help_text="Prefix to match incoming URIs (e.g., 's3://source-bucket/')",
    )

    input_profile = models.ForeignKey(
        StorageAccessProfile, on_delete=models.CASCADE, related_name="input_mappings"
    )

    output_profile = models.ForeignKey(
        StorageAccessProfile, on_delete=models.CASCADE, related_name="output_mappings"
    )

    output_base_uri = models.CharField(
        max_length=500,
        help_text="Base URI where Zarr files will be written (e.g., 'file:///tmp/zarr/' or 's3://target-bucket/encoded/')",
    )

    def __str__(self):
        return f"{self.input_prefix} → {self.output_base_uri}"


class AudioFile(TimeStampedModel, StatusModel):
    """
    Represents a specific input audio file being tracked for encoding.
    Linked to a StorageMapping that defines both input and output contexts.
    """

    STATUS = Choices(
        "initializing", "queued", "encoding", "encoded", "exception_returned"
    )

    uri = models.CharField(
        max_length=512,
        help_text="Full fsspec-compatible input URI (e.g., 's3://bucket/audio.wav')",
    )

    storage_mapping = models.ForeignKey(
        "StorageMapping",
        on_delete=models.PROTECT,
        related_name="audio_files",
        help_text="Routing configuration used to encode this file",
    )

    zarr_uri = models.CharField(
        max_length=512,
        blank=True,
        null=True,
        help_text="Full output URI where the encoded Zarr archive was written",
    )

    def __str__(self):
        return self.uri

    class Meta:
        unique_together = (("uri", "storage_mapping"),)
