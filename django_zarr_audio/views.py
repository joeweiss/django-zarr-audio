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
from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET
from zarr_audio.encoder import AudioEncoder
from zarr_audio.reader import AudioReader

from .credentials import get_fs_from_env
from .models import AudioFile, StorageMapping
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
        response = HttpResponse(f"File is {e}. Retry later.", status=504)
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
    use_precomputed = request.GET.get("_precomputed", "true").lower() == "true"

    print("use_precomputed", use_precomputed)

    try:
        reader = get_or_create_encoded_audio_reader(uri)
        duration = end - start

        # Try to use precomputed spectrogram if available and requested
        if use_precomputed and reader.has_spectrogram:
            print("using precomputed spectro data")

            try:
                S_db = reader.read_spectrogram_array(
                    start_time=start, duration=duration
                )
                print(S_db)
                params = reader.get_spectrogram_params()

                if use_pillow:
                    image_io = generate_spectrogram_image_from_array_pillow(
                        S_db,
                        reader.samplerate,
                        start,
                        end,
                        n_fft=params["n_fft"],
                        hop_length=params["hop_length"],
                        top=int(request.GET.get("top", 10000)),
                        noise_reduction=request.GET.get(
                            "noise_reduction", "false"
                        ).lower()
                        == "true",
                    )
                else:
                    image_io = generate_spectrogram_image_from_array(
                        S_db,
                        reader.samplerate,
                        start,
                        end,
                        n_fft=params["n_fft"],
                        hop_length=params["hop_length"],
                        top=int(request.GET.get("top", 10000)),
                        noise_reduction=request.GET.get(
                            "noise_reduction", "false"
                        ).lower()
                        == "true",
                    )
            except Exception as e:
                # Fall back to computing on-demand if precomputed fails
                print(
                    f"⚠️ Failed to use precomputed spectrogram: {e}. Computing on-demand."
                )
                use_precomputed = False

        # Compute spectrogram on-demand if precomputed not available or failed
        if not use_precomputed or not reader.has_spectrogram:
            print("using on-demand spectro data")
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
                    noise_reduction=request.GET.get("noise_reduction", "false").lower()
                    == "true",
                )
            else:
                image_io = generate_spectrogram_image(
                    encoded_bytes,
                    start,
                    end,
                    n_fft=int(request.GET.get("n_fft", 2048)),
                    hop_length=int(request.GET.get("hop_length", 512)),
                    top=int(request.GET.get("top", 10000)),
                    noise_reduction=request.GET.get("noise_reduction", "false").lower()
                    == "true",
                )
    except PermissionError:
        return HttpResponseBadRequest("Unauthorized or unmapped URI prefix")
    except ValueError as e:
        return HttpResponseBadRequest(str(e))
    except RuntimeError as e:
        response = HttpResponse(f"File is {e}. Retry later.", status=504)
        response["Retry-After"] = "30"
        return response
    except Exception as e:
        return HttpResponseBadRequest(f"Error generating spectrogram: {e}")

    return FileResponse(image_io, content_type="image/jpeg")


plt.ioff()  # disable interactive mode


def generate_spectrogram_image_from_array_pillow(
    S_db: np.ndarray,
    samplerate: int,
    start_sec: float,
    end_sec: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    top: int = 10000,
    top_db: float = 68.0,
    dpi: int = 144,
    height: float = 3.0,
    seconds_per_inch: float = 1,
    noise_reduction: bool = False,
) -> io.BytesIO:
    """Generate spectrogram image from precomputed dB array using Pillow."""
    print("generate_spectrogram_image_from_array_pillow", hop_length, n_fft)

    # Calculate dimensions
    duration = end_sec - start_sec
    fig_width = duration / seconds_per_inch
    width_px = int(fig_width * dpi)
    height_px = int(height * dpi)

    # Apply noise reduction if requested
    if noise_reduction:
        noise_threshold = np.percentile(S_db, 50, axis=1, keepdims=True)
        mask = S_db > (noise_threshold + 6)
        S_db = np.where(mask, S_db, S_db.min())

    # Crop frequencies
    n_rows, n_cols = S_db.shape
    freqs = np.linspace(0, samplerate / 2, n_rows, dtype=np.float32)
    max_bin = max(np.searchsorted(freqs, top, "right") - 1, 0)
    out_db = S_db[: max_bin + 1]
    del freqs

    # Normalize to 0-255 range
    db_min, db_max = out_db.min(), out_db.max()
    if db_max > db_min:
        normalized = ((out_db - db_min) / (db_max - db_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(out_db, dtype=np.uint8)

    # Flip vertically (low freq at bottom)
    normalized = np.flipud(normalized)

    # Create PIL Image and resize
    img = Image.fromarray(normalized, mode="L")
    img = img.resize((width_px, height_px), Image.Resampling.BILINEAR)

    # Apply colormap (inferno)
    cmap = plt.get_cmap("inferno")
    img_array = np.array(img, dtype=np.float32) / 255.0
    colored = cmap(img_array)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")

    # Save to BytesIO
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf


def generate_spectrogram_image_from_array(
    S_db: np.ndarray,
    samplerate: int,
    start_sec: float,
    end_sec: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    top: int = 10000,
    dpi: int = 144,
    cmap: str = "inferno",
    top_db: float = 68.0,
    height: float = 3.0,
    seconds_per_inch: float = 1,
    noise_reduction: bool = False,
) -> io.BytesIO:
    """Generate spectrogram image from precomputed dB array using matplotlib."""
    print("generate_spectrogram_image_from_array", hop_length, n_fft)

    # Apply noise reduction if requested
    if noise_reduction:
        noise_threshold = np.percentile(S_db, 50, axis=1, keepdims=True)
        mask = S_db > (noise_threshold + 6)
        S_db = np.where(mask, S_db, S_db.min())

    # Crop or pad frequencies
    n_rows, n_cols = S_db.shape
    freqs = np.linspace(0, samplerate / 2, n_rows, dtype=np.float32)
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
    del freqs

    # Plot with Matplotlib
    duration = end_sec - start_sec
    fig_width = duration / seconds_per_inch
    top_khz = (top if top <= samplerate / 2 else samplerate / 2) / 1000.0

    fig = plt.figure(figsize=(fig_width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        out_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, top_khz],
        cmap=cmap,
    )
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    fig.clf()
    plt.close(fig)

    buf.seek(0)
    return buf


def generate_spectrogram_image_pillow(
    encoded_bytes,
    start_sec: float,
    end_sec: float,
    top: int = 10000,
    n_fft: int = 1024,
    hop_length: int = 24,
    window: str = "hann",
    top_db: float = 68.0,
    dpi: int = 144,
    height: float = 3.0,
    seconds_per_inch: float = 1,
    noise_reduction: bool = False,
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
        S_db = librosa.amplitude_to_db(mag, ref=np.max(mag), top_db=top_db)
    else:
        S_db = librosa.amplitude_to_db(mag, ref=np.max(mag))
    del mag

    # 3.5) Noise reduction
    if noise_reduction:
        noise_threshold = np.percentile(S_db, 50, axis=1, keepdims=True)
        mask = S_db > (noise_threshold + 6)
        S_db = np.where(mask, S_db, S_db.min())

    # 4) Crop frequencies
    n_rows, n_cols = S_db.shape
    freqs = np.linspace(0, sr / 2, n_rows, dtype=np.float32)
    max_bin = max(np.searchsorted(freqs, top, "right") - 1, 0)
    out_db = S_db[: max_bin + 1]
    del S_db, freqs

    # 5) Normalize to 0-255 range
    db_min, db_max = out_db.min(), out_db.max()
    if db_max > db_min:
        normalized = ((out_db - db_min) / (db_max - db_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(out_db, dtype=np.uint8)

    # Flip vertically (low freq at bottom)
    normalized = np.flipud(normalized)

    # 6) Create PIL Image and resize
    img = Image.fromarray(normalized, mode="L")
    img = img.resize((width_px, height_px), Image.Resampling.BILINEAR)

    # 7) Apply colormap (inferno)
    # Convert grayscale to RGB using matplotlib's inferno colormap
    cmap = plt.get_cmap("inferno")
    # Apply colormap: normalize values to 0-1, then map to RGB
    img_array = np.array(img, dtype=np.float32) / 255.0
    colored = cmap(img_array)
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
    cmap: str = "inferno",
    top_db: float = 68.0,
    height: float = 3.0,
    seconds_per_inch: float = 1,
    noise_reduction: bool = False,
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
        S_db = librosa.amplitude_to_db(mag, ref=np.max(mag), top_db=top_db)
    else:
        S_db = librosa.amplitude_to_db(mag, ref=np.max(mag))
    del mag

    # 3.5) Noise reduction (mostly unused for now)
    if noise_reduction:
        # Adaptive noise gating: suppress values near the noise floor per frequency
        noise_threshold = np.percentile(S_db, 50, axis=1, keepdims=True)
        # Create mask where signal is above threshold + margin
        mask = S_db > (noise_threshold + 6)  # 6 dB above noise floor
        # Zero out noise, keep signals
        S_db = np.where(mask, S_db, S_db.min())

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
    ax.imshow(
        out_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, top_khz],
        cmap=cmap,
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
            audio_file.status = AudioFile.STATUS.encoded
            audio_file.zarr_uri = zarr_uri
            audio_file.save()
        except Exception as e:
            audio_file.status = AudioFile.STATUS.exception_returned
            audio_file.save()
            raise RuntimeError(f"Encoding failed: {e}")

    return AudioReader(zarr_uri, storage_options=fs_output.storage_options)
