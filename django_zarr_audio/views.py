import os
import tempfile
from urllib.parse import urlparse

from django.conf import settings
from django.http import FileResponse, HttpResponse, HttpResponseBadRequest, Http404
from django.shortcuts import render
from django.views.decorators.http import require_GET
from zarr_audio.encoder import AudioEncoder
from zarr_audio.reader import AudioReader

from .credentials import get_fs_from_env
from .models import AudioFile, StorageMapping
from .tasks import run_zarr_encoding
from .utils import get_output_uri, get_storage_mapping_for_uri
from django.contrib.auth.decorators import login_required


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


@login_required
def list_fsspec_files_view(request):
    if not getattr(settings, "DJZA_ENABLE_LISTING_VIEW", False):
        raise Http404("Listing view is disabled.")
    default_extensions = "wav,flac"

    if request.method == "GET":
        return render(
            request,
            "django_zarr_audio/list_fsspec_files.html",
            {
                "default_extensions": default_extensions,
            },
        )

    uri = request.POST.get("uri")
    extensions_input = request.POST.get("extensions", default_extensions)
    recursive = request.POST.get("recursive") == "on"

    if not uri:
        return HttpResponseBadRequest("Missing URI.")

    extensions = {
        (
            ext.strip().lower()
            if ext.strip().startswith(".")
            else f".{ext.strip().lower()}"
        )
        for ext in extensions_input.split(",")
        if ext.strip()
    }

    try:
        mapping = next(
            m
            for m in StorageMapping.objects.select_related("input_profile")
            if uri.startswith(m.input_prefix)
        )
    except StopIteration:
        return render(
            request,
            "django_zarr_audio/list_fsspec_files.html",
            {
                "error": "No matching StorageMapping found for the provided URI.",
                "uri": uri,
                "extensions": extensions_input,
                "recursive": recursive,
                "default_extensions": default_extensions,
            },
        )

    try:
        fs = get_fs_from_env(
            label=mapping.input_profile.credentials_label,
            backend=mapping.input_profile.backend,
        )

        if not uri.startswith(mapping.input_prefix):
            raise ValueError("Provided URI is outside the mapped input_prefix")

        # Reject potentially unsafe path traversal attempts
        parsed = urlparse(uri)
        if ".." in parsed.path.split("/"):
            raise ValueError("Unsafe path: '..' not allowed in URI")

        # Normalize and glob
        normalized_uri = uri.rstrip("/")
        pattern = f"{normalized_uri}/**/*" if recursive else f"{normalized_uri}/*"

        all_files = fs.glob(pattern)

        raw_protocol = fs.protocol
        if isinstance(raw_protocol, (tuple, list)):
            protocol = raw_protocol[0]
        else:
            protocol = raw_protocol

        matched_files = []

        for f in all_files:
            if protocol == "file":
                # Ensure correct triple-slash form
                full_uri = f"file://{f}" if f.startswith("/") else f"file:///{f}"
            else:
                full_uri = f"{protocol}://{f}"

            if uri.startswith(mapping.input_prefix) and full_uri.lower().endswith(
                tuple(extensions)
            ):
                matched_files.append(full_uri)

        matched_files.sort()

        MAX_FILES = getattr(settings, "DJZA_MAX_LISTED_FILES", 200)

        is_truncated = False
        if MAX_FILES is not None and len(matched_files) > MAX_FILES:
            is_truncated = True
            matched_files = matched_files[:MAX_FILES]

    except Exception as e:
        return render(
            request,
            "django_zarr_audio/list_fsspec_files.html",
            {
                "error": f"Error accessing files: {e}",
                "uri": uri,
                "extensions": extensions_input,
                "recursive": recursive,
                "default_extensions": default_extensions,
            },
        )

    return render(
        request,
        "django_zarr_audio/list_fsspec_files.html",
        {
            "files": matched_files,
            "uri": uri,
            "extensions": extensions_input,
            "recursive": recursive,
            "default_extensions": default_extensions,
            "is_truncated": is_truncated,
        },
    )
