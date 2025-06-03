import tempfile
from django.http import FileResponse, HttpResponse, HttpResponseBadRequest
from django.views.decorators.http import require_GET
from django.conf import settings

from .models import AudioFile, StorageMapping
from .credentials import get_fs_from_env
from .tasks import run_zarr_encoding
from .utils import get_output_uri, get_storage_mapping_for_uri
from zarr_audio.encoder import AudioEncoder
from zarr_audio.reader import AudioReader
import os


def health_check(request):
    return HttpResponse("OK", status=200)


def get_matching_mapping(uri):
    """Find the StorageMapping whose input_prefix matches the given URI."""
    for mapping in StorageMapping.objects.filter(status="active"):
        if uri.startswith(mapping.input_prefix):
            return mapping
    return None


class DeletingFileResponse(FileResponse):
    """
    A FileResponse that deletes the temporary file after the response is closed.
    """

    def __init__(self, tmp_file, *args, **kwargs):
        self._tmp_path = tmp_file.name
        super().__init__(tmp_file, *args, **kwargs)

    def close(self):
        super().close()
        try:
            os.remove(self._tmp_path)
        except FileNotFoundError:
            pass


@require_GET
def audio_proxy_view(request):
    uri = request.GET.get("uri")
    try:
        start = float(request.GET.get("start", 0))
        end = float(request.GET.get("end", start + 5))
    except (TypeError, ValueError):
        return HttpResponseBadRequest("Invalid 'start' or 'end' parameter")

    if not uri:
        return HttpResponseBadRequest("Missing 'uri' parameter")

    mapping = get_storage_mapping_for_uri(uri)
    if not mapping:
        return HttpResponseBadRequest("Unauthorized or unmapped URI prefix")

    fs_input = get_fs_from_env(
        mapping.input_profile.credentials_label, mapping.input_profile.backend
    )
    fs_output = get_fs_from_env(
        mapping.output_profile.credentials_label, mapping.output_profile.backend
    )

    try:
        zarr_uri = get_output_uri(uri, base_uri=mapping.output_base_uri)
    except ValueError as e:
        return HttpResponseBadRequest(str(e))

    audio_file, created = AudioFile.objects.get_or_create(
        uri=uri,
        storage_mapping=mapping,
        defaults={
            "status": AudioFile.STATUS.initializing,
            "zarr_uri": zarr_uri,
        },
    )

    if audio_file.status == AudioFile.STATUS.encoding:
        response = HttpResponse("File is encoding. Retry later.", status=504)
        response["Retry-After"] = "30"
        return response

    if audio_file.status == AudioFile.STATUS.queued:
        response = HttpResponse("File is queued for encoding. Retry later.", status=504)
        response["Retry-After"] = "30"
        return response

    if not fs_output.exists(zarr_uri):
        try:
            info = fs_input.info(uri)
            size = info["size"]
        except Exception as e:
            return HttpResponseBadRequest(f"Error reading file metadata: {e}")

        max_size = getattr(
            settings, "DJZA_MAX_IMMEDIATE_ENCODE_SIZE_BYTES", 100_000_000
        )

        if size > max_size:
            if created or audio_file.status == AudioFile.STATUS.initializing:
                audio_file.status = AudioFile.STATUS.queued
                audio_file.save()
                run_zarr_encoding(audio_file.id)
            response = HttpResponse(
                "File too large; queued for encoding. Retry later.", status=504
            )
            response["Retry-After"] = "30"
            return response

        # Small file: encode now
        try:
            chunk_duration = getattr(settings, "DJZA_ZARR_AUDIO_CHUNK_DURATION", 10)
            encoder = AudioEncoder(
                input_uri=uri,
                output_uri=zarr_uri,
                storage_options=fs_output.storage_options,
                chunk_duration=chunk_duration,
            )
            encoder.encode()
            audio_file.status = AudioFile.STATUS.encoded
            audio_file.zarr_uri = zarr_uri
            audio_file.save()
        except Exception as e:
            audio_file.status = AudioFile.STATUS.exception_returned
            audio_file.save()
            return HttpResponseBadRequest(f"Encoding failed: {e}")

    # Serve segment
    try:
        reader = AudioReader(zarr_uri, storage_options=fs_output.storage_options)
        duration = end - start
        encoded_bytes = reader.read_encoded(
            start_time=start, duration=duration, format="flac"
        )
    except KeyError:
        response = HttpResponse("File is encoding. Retry later.", status=504)
        response["Retry-After"] = "30"
        return response
    except Exception as e:
        return HttpResponseBadRequest(f"Error reading encoded segment: {e}")

    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        tmp_file.write(encoded_bytes)
        tmp_file.flush()
        tmp_file.seek(0)
    except Exception as e:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
        raise e

    return DeletingFileResponse(tmp_file, content_type="audio/flac")
