import numpy as np
import soundfile as sf
import tempfile
import os
import pathlib
import pytest
from django.urls import reverse

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
def test_audio_proxy_view_encodes_file_uri(client, settings, tmp_path):
    sr = 48000
    duration = 60  # seconds
    start_time = 2
    end_time = 7

    # Generate known test signal in range [-1, 1] for PCM_16
    samples = sr * duration
    expected_audio = (np.random.uniform(-1.0, 1.0, samples) * 0.99).astype(np.float32)

    input_path = tmp_path / "test_input.wav"
    output_base = tmp_path / "out"

    sf.write(input_path, expected_audio, samplerate=sr, subtype="PCM_16")
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

    uri = f"file://{input_path}"
    response = client.get(
        reverse("zap:audio-proxy"),
        {"uri": uri, "start": str(start_time), "end": str(end_time)},
    )

    assert response.status_code == 200
    assert response["Content-Type"] == "audio/flac"

    # Save the response to a temp file
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
        for chunk in response.streaming_content:
            f.write(chunk)
        flac_path = f.name

    # Decode the returned audio
    returned_audio, returned_sr = sf.read(flac_path)
    os.remove(flac_path)

    # Extract expected slice and normalize from int16 scale
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    target_audio = expected_audio[start_sample:end_sample]

    # Ensure both are 2D for comparison
    if returned_audio.ndim == 1:
        returned_audio = returned_audio[:, np.newaxis]
        target_audio = target_audio[:, np.newaxis]

    assert returned_sr == sr
    assert returned_audio.shape == target_audio.shape
    assert np.allclose(returned_audio, target_audio, rtol=1e-4, atol=1e-4)

    audio_file = AudioFile.objects.last()
    clean_up_zarr(audio_file)
    audio_file.delete()
