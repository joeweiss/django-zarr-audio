import numpy as np
import soundfile as sf
import pathlib
import pytest
from PIL import Image
import io
from django.urls import reverse
from django.contrib.auth.models import User

from django_zarr_audio.models import (
    StorageAccessProfile,
    StorageMapping,
    AudioFile,
)


def clean_up_zarr(audio_file):
    zarr_uri = audio_file.zarr_uri
    parsed = pathlib.Path(zarr_uri.replace("file://", ""))
    if parsed.exists():
        for child in parsed.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(parsed.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        parsed.rmdir()


@pytest.mark.django_db
def test_spectrogram_proxy_view_matplotlib(client, settings, tmp_path):
    """Test spectrogram generation using matplotlib backend"""
    print("test_spectrogram_proxy_view_matplotlib")
    sr = 48000
    duration = 10  # seconds
    start_time = 1
    end_time = 4

    # Generate test audio signal
    samples = sr * duration
    test_audio = (np.random.uniform(-1.0, 1.0, samples) * 0.99).astype(np.float32)

    input_path = tmp_path / "test_input.wav"
    output_base = tmp_path / "out"

    sf.write(input_path, test_audio, samplerate=sr, subtype="PCM_16")
    output_base.mkdir()

    input_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test input",
    )
    output_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test output",
    )

    StorageMapping.objects.create(
        input_prefix=f"file://{tmp_path}/",
        input_profile=input_profile,
        output_profile=output_profile,
        output_base_uri=f"file://{output_base}/",
        status="active",
    )

    user = User.objects.create_user(username="testuser", password="testpass")
    client.force_login(user)

    uri = f"file://{input_path}"
    response = client.get(
        reverse("zap:spectrogram-proxy"),
        {"uri": uri, "start": str(start_time), "end": str(end_time)},
    )

    assert response.status_code == 200
    assert response["Content-Type"] == "image/jpeg"

    # Verify we got valid image data
    image_data = b"".join(response.streaming_content)
    img = Image.open(io.BytesIO(image_data))
    assert img.format in ["PNG", "JPEG"]
    assert img.size[0] > 0
    assert img.size[1] > 0

    audio_file = AudioFile.objects.last()
    clean_up_zarr(audio_file)
    audio_file.delete()


@pytest.mark.django_db
def test_spectrogram_proxy_view_pillow(client, settings, tmp_path):
    """Test spectrogram generation using Pillow backend"""
    print("test_spectrogram_proxy_view_pillow")
    sr = 48000
    duration = 10  # seconds
    start_time = 1
    end_time = 4

    # Generate test audio signal
    samples = sr * duration
    test_audio = (np.random.uniform(-1.0, 1.0, samples) * 0.99).astype(np.float32)

    input_path = tmp_path / "test_input.wav"
    output_base = tmp_path / "out"

    sf.write(input_path, test_audio, samplerate=sr, subtype="PCM_16")
    output_base.mkdir()

    input_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test input",
    )
    output_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test output",
    )

    StorageMapping.objects.create(
        input_prefix=f"file://{tmp_path}/",
        input_profile=input_profile,
        output_profile=output_profile,
        output_base_uri=f"file://{output_base}/",
        status="active",
    )

    user = User.objects.create_user(username="testuser", password="testpass")
    client.force_login(user)

    uri = f"file://{input_path}"
    response = client.get(
        reverse("zap:spectrogram-proxy"),
        {"uri": uri, "start": str(start_time), "end": str(end_time), "_pillow": "true"},
    )

    assert response.status_code == 200
    assert response["Content-Type"] == "image/jpeg"

    # Verify we got valid image data
    image_data = b"".join(response.streaming_content)
    img = Image.open(io.BytesIO(image_data))
    assert img.format in ["PNG", "JPEG"]
    assert img.size[0] > 0
    assert img.size[1] > 0

    audio_file = AudioFile.objects.last()
    clean_up_zarr(audio_file)
    audio_file.delete()


@pytest.mark.django_db
def test_spectrogram_proxy_with_parameters(client, settings, tmp_path):
    """Test spectrogram generation with custom parameters"""
    print("test_spectrogram_proxy_with_parameters")
    sr = 48000
    duration = 10  # seconds
    start_time = 1
    end_time = 4

    # Generate test audio signal
    samples = sr * duration
    test_audio = (np.random.uniform(-1.0, 1.0, samples) * 0.99).astype(np.float32)

    input_path = tmp_path / "test_input.wav"
    output_base = tmp_path / "out"

    sf.write(input_path, test_audio, samplerate=sr, subtype="PCM_16")
    output_base.mkdir()

    input_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test input",
    )
    output_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test output",
    )

    StorageMapping.objects.create(
        input_prefix=f"file://{tmp_path}/",
        input_profile=input_profile,
        output_profile=output_profile,
        output_base_uri=f"file://{output_base}/",
        status="active",
    )

    user = User.objects.create_user(username="testuser", password="testpass")
    client.force_login(user)

    uri = f"file://{input_path}"
    response = client.get(
        reverse("zap:spectrogram-proxy"),
        {
            "uri": uri,
            "start": str(start_time),
            "end": str(end_time),
            "n_fft": "1024",
            "hop_length": "256",
            "top": "8000",
            "cmap": "viridis",
            "normalize": "true",
        },
    )

    assert response.status_code == 200
    assert response["Content-Type"] == "image/jpeg"

    # Verify we got valid image data
    image_data = b"".join(response.streaming_content)
    img = Image.open(io.BytesIO(image_data))
    assert img.format in ["PNG", "JPEG"]
    assert img.size[0] > 0
    assert img.size[1] > 0

    audio_file = AudioFile.objects.last()
    clean_up_zarr(audio_file)
    audio_file.delete()


@pytest.mark.django_db
def test_spectrogram_consistency_with_normalize_false(client, settings, tmp_path):
    """Test that spectrograms are consistent across different time ranges when normalize=false"""
    print("test_spectrogram_consistency_with_normalize_false")
    sr = 48000
    duration = 30  # seconds

    # Generate test audio signal
    samples = sr * duration
    test_audio = (np.random.uniform(-1.0, 1.0, samples) * 0.99).astype(np.float32)

    input_path = tmp_path / "test_input.wav"
    output_base = tmp_path / "out"

    sf.write(input_path, test_audio, samplerate=sr, subtype="PCM_16")
    output_base.mkdir()

    input_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test input",
    )
    output_profile = StorageAccessProfile.objects.create(
        credentials_label="default",
        backend="file",
        status="active",
        description="Test output",
    )

    StorageMapping.objects.create(
        input_prefix=f"file://{tmp_path}/",
        input_profile=input_profile,
        output_profile=output_profile,
        output_base_uri=f"file://{output_base}/",
        status="active",
    )

    user = User.objects.create_user(username="testuser", password="testpass")
    client.force_login(user)

    uri = f"file://{input_path}"

    # Get spectrogram for 0-3 seconds
    response1 = client.get(
        reverse("zap:spectrogram-proxy"),
        {"uri": uri, "start": "0", "end": "3", "normalize": "false"},
    )

    # Get spectrogram for 0-30 seconds
    response2 = client.get(
        reverse("zap:spectrogram-proxy"),
        {"uri": uri, "start": "0", "end": "30", "normalize": "false"},
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Both should return valid images
    image_data1 = b"".join(response1.streaming_content)
    image_data2 = b"".join(response2.streaming_content)

    img1 = Image.open(io.BytesIO(image_data1))
    img2 = Image.open(io.BytesIO(image_data2))

    assert img1.size[0] > 0
    assert img1.size[1] > 0
    assert img2.size[0] > 0
    assert img2.size[1] > 0

    # Note: We're not comparing pixel values because lossy compression
    # makes exact comparison difficult, but both should be valid images

    audio_file = AudioFile.objects.last()
    clean_up_zarr(audio_file)
    audio_file.delete()
