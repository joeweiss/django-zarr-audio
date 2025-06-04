import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from django_zarr_audio.models import StorageAccessProfile, StorageMapping


@pytest.mark.django_db()
def test_list_view_404_by_default(client, settings):
    # Create and log in a test user
    User.objects.create_user(username="testuser", password="testpass")
    logged_in = client.login(username="testuser", password="testpass")
    assert logged_in is True

    # Setup test credentials
    profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="s3",
        status="active",
        description="Test listing access",
    )

    StorageMapping.objects.create(
        input_prefix="s3://my-bucket/",
        input_profile=profile,
        output_profile=profile,
        output_base_uri="s3://my-bucket/ZAP/",
        status="active",
    )

    uri = "s3://my-bucket/HOME/"
    url = reverse("zap:list-fsspec-files")

    response = client.post(
        url,
        {
            "uri": uri,
            "extensions": "wav,flac",
            "recursive": "on",
        },
    )

    # Confirm that view is 404 without DJZA_ENABLE_LISTING_VIEW=True
    assert response.status_code == 404
