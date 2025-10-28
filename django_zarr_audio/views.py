import io
import os
import tempfile
from urllib.parse import urlparse

import librosa


import numpy as np
import soundfile as sf
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from zarr_audio.encoder import AudioEncoder
from zarr_audio.reader import AudioReader

from .credentials import get_fs_from_env
from .models import AudioFile, StorageAccessProfile, StorageMapping
from .tasks import run_zarr_encoding
from .utils import get_output_uri, get_storage_mapping_for_uri


import matplotlib

matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image


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


@login_required
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

    try:
        reader = get_or_create_encoded_audio_reader(uri)
        duration = end - start
        encoded_bytes = reader.read_encoded(
            start_time=start, duration=duration, format="flac"
        )
    except PermissionError:
        return HttpResponseBadRequest("Unauthorized or unmapped URI prefix")
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    except RuntimeError as e:
        error_msg = str(e)
        # Only return 202 for queued/encoding status, not for actual errors
        if error_msg in ("queued", "encoding"):
            response = HttpResponse(f"File is {e}. Retry later.", status=202)
            response["Retry-After"] = "5"
            return response
        else:
            # Actual encoding error
            return HttpResponseBadRequest(f"Encoding error: {e}")
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
def browse_storage_view(request, mapping_id=None):
    """
    Hierarchical file browser for StorageMappings.
    - No mapping_id: show all available mappings
    - With mapping_id: browse directories and files under that mapping
    """
    if not getattr(settings, "DJZA_ENABLE_LISTING_VIEW", False):
        raise Http404("Listing view is disabled.")

    default_extensions = "wav,flac"

    # If no mapping_id, show list of all active mappings
    if mapping_id is None:
        mappings = StorageMapping.objects.filter(status="active").select_related(
            "input_profile", "output_profile"
        )
        return render(
            request,
            "django_zarr_audio/browse_storage.html",
            {
                "mappings": mappings,
                "default_extensions": default_extensions,
            },
        )

    # Get the specific mapping
    try:
        mapping = StorageMapping.objects.select_related(
            "input_profile", "output_profile"
        ).get(id=mapping_id, status="active")
    except StorageMapping.DoesNotExist:
        raise Http404("Storage mapping not found or inactive.")

    # Get path from query string (relative to mapping's input_prefix)
    relative_path = request.GET.get("path", "").strip()
    extensions_input = request.GET.get("extensions", default_extensions)

    # Parse extensions
    extensions = {
        (
            ext.strip().lower()
            if ext.strip().startswith(".")
            else f".{ext.strip().lower()}"
        )
        for ext in extensions_input.split(",")
        if ext.strip()
    }

    # Security: reject path traversal attempts
    if ".." in relative_path.split("/"):
        return HttpResponseBadRequest("Unsafe path: '..' not allowed")

    # Build full URI
    base_uri = mapping.input_prefix.rstrip("/")
    if relative_path:
        current_uri = f"{base_uri}/{relative_path.lstrip('/')}"
    else:
        current_uri = base_uri

    # Build breadcrumbs
    breadcrumbs = [
        {
            "name": "Mappings",
            "url": request.path.split("/browse-storage/")[0] + "/browse-storage/",
        }
    ]
    breadcrumbs.append({"name": f"{mapping.input_prefix}", "url": f"?path="})

    if relative_path:
        path_parts = relative_path.strip("/").split("/")
        accumulated_path = ""
        for part in path_parts:
            accumulated_path = f"{accumulated_path}/{part}".lstrip("/")
            breadcrumbs.append({"name": part, "url": f"?path={accumulated_path}"})

    try:
        fs = get_fs_from_env(
            label=mapping.input_profile.credentials_label,
            backend=mapping.input_profile.backend,
        )

        # List contents of current directory
        normalized_uri = current_uri.rstrip("/")

        # Get protocol
        raw_protocol = fs.protocol
        if isinstance(raw_protocol, (tuple, list)):
            protocol = raw_protocol[0]
        else:
            protocol = raw_protocol

        # List direct children only (not recursive)
        try:
            all_items = fs.ls(normalized_uri, detail=True)
        except FileNotFoundError:
            all_items = []

        directories = []
        files = []

        for item in all_items:
            # Handle different fsspec backends' response formats
            if isinstance(item, dict):
                item_path = item.get("name", item.get("Key", ""))
                item_type = item.get("type", item.get("StorageClass", "file"))
            else:
                item_path = item
                item_type = "file"

            # Build full URI
            if protocol == "file":
                full_uri = (
                    f"file://{item_path}"
                    if item_path.startswith("/")
                    else f"file:///{item_path}"
                )
            else:
                full_uri = f"{protocol}://{item_path}"

            # Determine if directory or file
            if item_type == "directory":
                # Calculate relative path for this directory
                if full_uri.startswith(base_uri):
                    dir_relative = full_uri[len(base_uri) :].lstrip("/")
                else:
                    dir_relative = item_path.split("/")[-1]

                directories.append(
                    {
                        "name": item_path.split("/")[-1] or item_path.split("/")[-2],
                        "relative_path": dir_relative,
                    }
                )
            else:
                # Check if matches extensions
                if full_uri.lower().endswith(tuple(extensions)):
                    files.append(full_uri)

        directories.sort(key=lambda x: x["name"])
        files.sort()

        # Apply file limit
        MAX_FILES = getattr(settings, "DJZA_MAX_LISTED_FILES", 200)
        is_truncated = False
        if MAX_FILES is not None and len(files) > MAX_FILES:
            is_truncated = True
            files = files[:MAX_FILES]

        # Get encoding status and duration for files
        from .models import AudioFile
        file_uris = {f for f in files}
        audio_files = AudioFile.objects.filter(uri__in=file_uris).only('uri', 'status', 'duration_seconds')
        file_info_map = {af.uri: {'status': af.status, 'duration': af.duration_seconds} for af in audio_files}

        # Attach status and duration to each file
        def format_duration(seconds):
            if seconds is None:
                return None
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes:02d}:{secs:02d}"

        files_with_status = [
            {
                'uri': file_uri,
                'name': file_uri.split('/')[-1],
                'status': file_info_map.get(file_uri, {}).get('status'),
                'duration': format_duration(file_info_map.get(file_uri, {}).get('duration')),
            }
            for file_uri in files
        ]

    except Exception as e:
        return render(
            request,
            "django_zarr_audio/browse_storage.html",
            {
                "error": f"Error accessing storage: {e}",
                "mapping": mapping,
                "breadcrumbs": breadcrumbs,
                "extensions": extensions_input,
                "default_extensions": default_extensions,
            },
        )

    return render(
        request,
        "django_zarr_audio/browse_storage.html",
        {
            "mapping": mapping,
            "breadcrumbs": breadcrumbs,
            "directories": directories,
            "files": files_with_status,
            "current_path": relative_path,
            "extensions": extensions_input,
            "default_extensions": default_extensions,
            "is_truncated": is_truncated,
        },
    )


@login_required
def add_storage_mapping_view(request):
    """
    Form to create a new StorageMapping and associated StorageAccessProfiles if needed.
    """
    if not getattr(settings, "DJZA_ENABLE_LISTING_VIEW", False):
        raise Http404("Listing view is disabled.")

    if request.method == "GET":
        # Get existing profiles for the form
        profiles = StorageAccessProfile.objects.filter(status="active")
        return render(
            request,
            "django_zarr_audio/add_storage_mapping.html",
            {
                "profiles": profiles,
            },
        )

    # POST: Create new mapping
    input_prefix = request.POST.get("input_prefix", "").strip()
    output_base_uri = request.POST.get("output_base_uri", "").strip()

    # Input profile: either select existing or create new
    input_profile_id = request.POST.get("input_profile_id")
    create_new_input = request.POST.get("create_new_input") == "on"

    # Output profile: either select existing or create new
    output_profile_id = request.POST.get("output_profile_id")
    create_new_output = request.POST.get("create_new_output") == "on"

    errors = []

    # Validation
    if not input_prefix:
        errors.append("Input prefix is required.")
    if not output_base_uri:
        errors.append("Output base URI is required.")

    # Handle input profile
    if create_new_input:
        input_creds = request.POST.get("input_credentials_label", "").strip()
        input_backend = request.POST.get("input_backend", "").strip()
        input_desc = request.POST.get("input_description", "").strip()

        if not input_creds:
            errors.append("Input credentials label is required.")
        if not input_backend:
            errors.append("Input backend is required.")

        if not errors:
            input_profile = StorageAccessProfile.objects.create(
                credentials_label=input_creds,
                backend=input_backend,
                description=input_desc,
                status="active",
            )
    else:
        if not input_profile_id:
            errors.append("Input profile must be selected or created.")
        else:
            try:
                input_profile = StorageAccessProfile.objects.get(id=input_profile_id)
            except StorageAccessProfile.DoesNotExist:
                errors.append("Selected input profile not found.")

    # Handle output profile
    if create_new_output:
        output_creds = request.POST.get("output_credentials_label", "").strip()
        output_backend = request.POST.get("output_backend", "").strip()
        output_desc = request.POST.get("output_description", "").strip()

        if not output_creds:
            errors.append("Output credentials label is required.")
        if not output_backend:
            errors.append("Output backend is required.")

        if not errors:
            output_profile = StorageAccessProfile.objects.create(
                credentials_label=output_creds,
                backend=output_backend,
                description=output_desc,
                status="active",
            )
    else:
        if not output_profile_id:
            errors.append("Output profile must be selected or created.")
        else:
            try:
                output_profile = StorageAccessProfile.objects.get(id=output_profile_id)
            except StorageAccessProfile.DoesNotExist:
                errors.append("Selected output profile not found.")

    if errors:
        profiles = StorageAccessProfile.objects.filter(status="active")
        return render(
            request,
            "django_zarr_audio/add_storage_mapping.html",
            {
                "profiles": profiles,
                "errors": errors,
                "form_data": request.POST,
            },
        )

    # Validate input storage access
    try:
        fs_input = get_fs_from_env(
            label=input_profile.credentials_label,
            backend=input_profile.backend,
        )
        # Try to access the input prefix
        parsed = urlparse(input_prefix)
        path_to_check = parsed.path if parsed.scheme else input_prefix

        # Remove protocol prefix for fs operations
        if input_profile.backend == "file":
            check_path = path_to_check
        else:
            # For s3://bucket/path, we need just bucket/path
            check_path = parsed.netloc + parsed.path if parsed.netloc else path_to_check

        # Try to list the directory to validate access
        try:
            fs_input.ls(check_path.rstrip("/"), detail=False)
        except FileNotFoundError:
            # Directory doesn't exist yet - that's okay, but check parent exists or we can create it
            pass
        except Exception as e:
            errors.append(f"Cannot access input prefix: {e}")
    except Exception as e:
        errors.append(f"Invalid input storage configuration: {e}")

    # Validate output storage access
    try:
        fs_output = get_fs_from_env(
            label=output_profile.credentials_label,
            backend=output_profile.backend,
        )
        # Try to access the output base URI
        parsed = urlparse(output_base_uri)
        path_to_check = parsed.path if parsed.scheme else output_base_uri

        # Remove protocol prefix for fs operations
        if output_profile.backend == "file":
            check_path = path_to_check
        else:
            check_path = parsed.netloc + parsed.path if parsed.netloc else path_to_check

        # Try to list/create the directory to validate write access
        try:
            fs_output.ls(check_path.rstrip("/"), detail=False)
        except FileNotFoundError:
            # Try to create the directory to test write access
            try:
                fs_output.makedirs(check_path.rstrip("/"), exist_ok=True)
            except Exception as e:
                errors.append(
                    f"Cannot create output directory (check write permissions): {e}"
                )
        except Exception as e:
            errors.append(f"Cannot access output base URI: {e}")
    except Exception as e:
        errors.append(f"Invalid output storage configuration: {e}")

    # If validation errors occurred, show them
    if errors:
        profiles = StorageAccessProfile.objects.filter(status="active")
        return render(
            request,
            "django_zarr_audio/add_storage_mapping.html",
            {
                "profiles": profiles,
                "errors": errors,
                "form_data": request.POST,
            },
        )

    # Create the mapping
    try:
        StorageMapping.objects.create(
            input_prefix=input_prefix,
            input_profile=input_profile,
            output_profile=output_profile,
            output_base_uri=output_base_uri,
            status="active",
        )
        # Redirect to browse storage on success
        from django.shortcuts import redirect

        return redirect("zap:browse-storage")
    except Exception as e:
        profiles = StorageAccessProfile.objects.filter(status="active")
        return render(
            request,
            "django_zarr_audio/add_storage_mapping.html",
            {
                "profiles": profiles,
                "errors": [f"Error creating mapping: {e}"],
                "form_data": request.POST,
            },
        )


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


@require_GET
def spectrogram_proxy_view(request):
    uri = request.GET.get("uri")
    try:
        start = float(request.GET.get("start", 0))
        end = float(request.GET.get("end", start + 5))
    except (TypeError, ValueError):
        return HttpResponseBadRequest("Invalid 'start' or 'end' parameter")

    if not uri:
        return HttpResponseBadRequest("Missing 'uri' parameter")

    use_pillow = request.GET.get("_pillow", "false").lower() == "true"

    try:
        reader = get_or_create_encoded_audio_reader(uri)
        duration = end - start
        encoded_bytes = reader.read_encoded(
            start_time=start, duration=duration, format="flac"
        )

        if use_pillow:
            image_io = generate_spectrogram_image_pillow(
                encoded_bytes,
                start,
                end,
                n_fft=int(request.GET.get("n_fft", 2048)),
                hop_length=int(request.GET.get("hop_length", 512)),
                top=int(request.GET.get("top", 10000)),
                cmap=request.GET.get("cmap", "viridis"),
                normalize=request.GET.get("normalize", "false").lower() == "true",
            )
        else:
            image_io = generate_spectrogram_image(
                encoded_bytes,
                start,
                end,
                n_fft=int(request.GET.get("n_fft", 2048)),
                hop_length=int(request.GET.get("hop_length", 512)),
                top=int(request.GET.get("top", 10000)),
                cmap=request.GET.get("cmap", "viridis"),
                normalize=request.GET.get("normalize", "false").lower() == "true",
            )
    except PermissionError:
        return HttpResponseBadRequest("Unauthorized or unmapped URI prefix")
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    except RuntimeError as e:
        error_msg = str(e)
        # Only return 202 for queued/encoding status, not for actual errors
        if error_msg in ("queued", "encoding"):
            response = HttpResponse(f"File is {e}. Retry later.", status=202)
            response["Retry-After"] = "5"
            return response
        else:
            # Actual encoding error
            return HttpResponseBadRequest(f"Encoding error: {e}")
    except Exception as e:
        return HttpResponseBadRequest(f"Error generating spectrogram: {e}")

    return FileResponse(image_io, content_type="image/jpeg")


plt.ioff()  # disable interactive mode


def generate_spectrogram_image_pillow(
    encoded_bytes,
    start_sec: float,
    end_sec: float,
    top: int = 10000,
    n_fft: int = 1024,
    hop_length: int = 24,
    window: str = "hann",
    cmap: str = "viridis",
    top_db: float = 68.0,
    dpi: int = 144,
    height: float = 3.0,
    seconds_per_inch: float = 1,
    normalize: bool = False,
) -> io.BytesIO:
    """Generate spectrogram using Pillow for faster rendering."""
    print("generate_spectrogram_image_pillow", hop_length, n_fft)

    # 1) Load & downcast
    with io.BytesIO(encoded_bytes) as audio_io:
        y, sr = sf.read(audio_io)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1, dtype=np.float32)

    # Calculate dimensions based on audio duration
    duration = y.shape[0] / sr
    fig_width = duration / seconds_per_inch
    width_px = int(fig_width * dpi)
    height_px = int(height * dpi)

    # 2) STFT & magnitude
    win = (np.hanning(n_fft) if window == "hann" else np.kaiser(n_fft, beta=14)).astype(
        np.float32
    )
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=win)
    mag = np.abs(S, dtype=np.float32)
    del S, win

    # 3) dB conversion
    if top_db is not None:
        S_db = librosa.amplitude_to_db(mag, ref=1.0, top_db=top_db)
    else:
        S_db = librosa.amplitude_to_db(mag, ref=1.0)
    del mag

    # 4) Crop frequencies
    n_rows, n_cols = S_db.shape
    freqs = np.linspace(0, sr / 2, n_rows, dtype=np.float32)
    max_bin = max(np.searchsorted(freqs, top, "right") - 1, 0)
    out_db = S_db[: max_bin + 1]
    del S_db, freqs

    # 5) Normalize to 0-255 range
    if normalize:
        # Auto-scale: use actual min/max from this segment
        db_min, db_max = out_db.min(), out_db.max()
        if db_max > db_min:
            normalized = ((out_db - db_min) / (db_max - db_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(out_db, dtype=np.uint8)
    else:
        # Fixed range: -top_db to 0 dB for consistent visualization
        db_min, db_max = -top_db, 0.0
        normalized = np.clip((out_db - db_min) / (db_max - db_min) * 255, 0, 255).astype(
            np.uint8
        )

    # Flip vertically (low freq at bottom)
    normalized = np.flipud(normalized)

    # 6) Create PIL Image and resize
    img = Image.fromarray(normalized, mode="L")
    img = img.resize((width_px, height_px), Image.Resampling.BILINEAR)

    # 7) Apply colormap
    # Convert grayscale to RGB using matplotlib colormap
    cmap_obj = plt.get_cmap(cmap)
    # Apply colormap: normalize values to 0-1, then map to RGB
    img_array = np.array(img, dtype=np.float32) / 255.0
    colored = cmap_obj(img_array)
    # Convert to RGB (drop alpha channel) and scale to 0-255
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")

    # 8) Save to BytesIO
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf


def generate_spectrogram_image(
    encoded_bytes,
    start_sec: float,
    end_sec: float,
    top: int = 10000,
    dpi: int = 144,
    n_fft: int = 1024,
    hop_length: int = 24,
    window: str = "hann",
    cmap: str = "viridis",
    top_db: float = 68.0,
    height: float = 3.0,
    seconds_per_inch: float = 1,
    normalize: bool = False,
) -> io.BytesIO:
    print("generate_spectrogram_image", hop_length, n_fft)

    # 1) Load & downcast
    with io.BytesIO(encoded_bytes) as audio_io:
        y, sr = sf.read(audio_io)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        # mono-mix in float32
        y = np.mean(y, axis=1, dtype=np.float32)

    # 2) STFT & magnitude
    win = (np.hanning(n_fft) if window == "hann" else np.kaiser(n_fft, beta=14)).astype(
        np.float32
    )
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=win)
    mag = np.abs(S, dtype=np.float32)
    del S, win

    # 3) dB conversion
    if top_db is not None:
        S_db = librosa.amplitude_to_db(mag, ref=1.0, top_db=top_db)
    else:
        S_db = librosa.amplitude_to_db(mag, ref=1.0)
    del mag

    # 4) Crop or pad frequencies
    n_rows, n_cols = S_db.shape
    freqs = np.linspace(0, sr / 2, n_rows, dtype=np.float32)
    freq_res = freqs[1] - freqs[0]
    desired_rows = int(np.ceil(top / freq_res)) + 1

    if desired_rows > n_rows:
        floor = S_db.min()
        out_db = np.empty((desired_rows, n_cols), dtype=np.float32)
        out_db[:n_rows] = S_db
        out_db[n_rows:] = floor
    else:
        max_bin = max(np.searchsorted(freqs, top, "right") - 1, 0)
        out_db = S_db[: max_bin + 1]
    del S_db, freqs

    # 5) Plot with Matplotlib
    duration = y.shape[0] / sr
    fig_width = duration / seconds_per_inch
    top_khz = (top if top <= sr / 2 else sr / 2) / 1000.0

    fig = plt.figure(figsize=(fig_width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    if normalize:
        # Auto-scale: let imshow determine vmin/vmax from data
        ax.imshow(
            out_db,
            aspect="auto",
            origin="lower",
            extent=[0, duration, 0, top_khz],
            cmap=cmap,
        )
    else:
        # Fixed range: -top_db to 0 dB for consistent visualization
        ax.imshow(
            out_db,
            aspect="auto",
            origin="lower",
            extent=[0, duration, 0, top_khz],
            cmap=cmap,
            vmin=-top_db,
            vmax=0.0,
        )
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    # Clean up figure state immediately
    fig.clf()
    plt.close(fig)

    buf.seek(0)
    return buf


# Helper functions (common to both audio and spectrogram)


def get_or_create_audio_file(uri, mapping, zarr_uri):
    with transaction.atomic():
        try:
            # lock any existing row
            af = AudioFile.objects.select_for_update().get(
                uri=uri, storage_mapping=mapping
            )
            return af, False
        except AudioFile.DoesNotExist:
            # safe to insert now
            af = AudioFile.objects.create(
                uri=uri,
                storage_mapping=mapping,
                status=AudioFile.STATUS.initializing,
                zarr_uri=zarr_uri,
            )
            return af, True


def get_or_create_encoded_audio_reader(uri):
    mapping = get_storage_mapping_for_uri(uri)
    if not mapping:
        raise PermissionError("Unauthorized or unmapped URI prefix")

    fs_input = get_fs_from_env(
        mapping.input_profile.credentials_label, mapping.input_profile.backend
    )
    fs_output = get_fs_from_env(
        mapping.output_profile.credentials_label, mapping.output_profile.backend
    )

    try:
        zarr_uri = get_output_uri(uri, base_uri=mapping.output_base_uri)
    except ValueError as e:
        raise ValueError(str(e))

    audio_file, created = get_or_create_audio_file(uri, mapping, zarr_uri)

    if audio_file.status == AudioFile.STATUS.encoding:
        raise RuntimeError("encoding")

    if audio_file.status == AudioFile.STATUS.queued:
        raise RuntimeError("queued")

    if not fs_output.exists(zarr_uri):
        try:
            info = fs_input.info(uri)
            size = info["size"]
        except Exception as e:
            raise IOError(f"Error reading file metadata: {e}")

        max_size = getattr(
            settings, "DJZA_MAX_IMMEDIATE_ENCODE_SIZE_BYTES", 100_000_000
        )
        if size > max_size:
            if created or audio_file.status == AudioFile.STATUS.initializing:
                audio_file.status = AudioFile.STATUS.queued
                audio_file.save()
                run_zarr_encoding(audio_file.id)
            raise RuntimeError("queued")

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

            # Get duration from the encoded zarr file
            try:
                reader = AudioReader(zarr_uri, storage_options=fs_output.storage_options)
                info = reader.info()
                audio_file.duration_seconds = info.get('duration_sec')
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to get duration for {uri}: {e}")
                pass  # If we can't get duration, leave it null

            audio_file.status = AudioFile.STATUS.encoded
            audio_file.zarr_uri = zarr_uri
            audio_file.save()
        except Exception as e:
            audio_file.status = AudioFile.STATUS.exception_returned
            audio_file.save()
            error_msg = str(e)
            # Provide clearer error message for common issues
            if "Format not recognised" in error_msg or "does not appear to be an audio file" in error_msg:
                raise RuntimeError("File encoding failed. The file may be corrupted or the requested time range may be invalid.")
            else:
                raise RuntimeError(f"Encoding failed: {e}")
    else:
        # File is already encoded, update status and duration if needed
        reader = AudioReader(zarr_uri, storage_options=fs_output.storage_options)

        needs_update = False

        if audio_file.status != AudioFile.STATUS.encoded:
            audio_file.status = AudioFile.STATUS.encoded
            audio_file.zarr_uri = zarr_uri
            needs_update = True

        if audio_file.duration_seconds is None:
            try:
                info = reader.info()
                audio_file.duration_seconds = info.get('duration_sec')
                needs_update = True
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to get duration for {uri}: {e}")
                pass

        if needs_update:
            audio_file.save()

        return reader

    # Should not reach here, but return reader if we somehow do
    return AudioReader(zarr_uri, storage_options=fs_output.storage_options)


@login_required
def file_info_view(request):
    """
    Returns metadata about an audio file, including duration.
    GET params: uri
    Returns JSON with duration_seconds if available.
    """
    uri = request.GET.get("uri")
    if not uri:
        return JsonResponse({"error": "Missing uri parameter"}, status=400)

    mapping = get_storage_mapping_for_uri(uri)
    if not mapping:
        return JsonResponse({"error": "Unauthorized or unmapped URI prefix"}, status=403)

    try:
        audio_file = AudioFile.objects.get(uri=uri, storage_mapping=mapping)
        return JsonResponse({
            "uri": audio_file.uri,
            "status": audio_file.status,
            "duration_seconds": audio_file.duration_seconds,
        })
    except AudioFile.DoesNotExist:
        return JsonResponse({
            "uri": uri,
            "status": None,
            "duration_seconds": None,
        })
