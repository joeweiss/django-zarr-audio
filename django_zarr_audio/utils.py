import os
from pathlib import PurePosixPath

from fsspec.utils import infer_storage_options
from typing import Optional
from .models import StorageMapping


def get_storage_mapping_for_uri(uri: str) -> Optional[StorageMapping]:
    """
    Return the StorageMapping whose input_prefix matches the given URI.
    Chooses the longest matching prefix if multiple apply.
    """
    mappings = StorageMapping.objects.filter(
        input_prefix__isnull=False, status="active"
    )
    sorted_mappings = sorted(mappings, key=lambda m: len(m.input_prefix), reverse=True)

    for mapping in sorted_mappings:
        if uri.startswith(mapping.input_prefix):
            return mapping
    return None


def get_storage_options(label: str) -> dict:
    return {
        "key": os.getenv(f"DJZA_{label}_KEY"),
        "secret": os.getenv(f"DJZA_{label}_SECRET"),
        "client_kwargs": {
            "region_name": os.getenv(f"DJZA_{label}_REGION", "us-east-1"),
        },
    }


def safe_uri_join(base_uri: str, relative_path: str) -> str:
    if not base_uri.endswith("/"):
        base_uri += "/"
    return base_uri + relative_path.lstrip("/")


def get_output_uri(input_uri: str, base_uri: str) -> str:
    """
    Given an input URI and an output base URI, return the full Zarr output URI.

    - Appends `.zarr` suffix to the filename (preserving original extension)
    - Uses base_uri from StorageMapping to build full output path
    """
    if not base_uri:
        raise ValueError("Missing output_base_uri")

    parts = infer_storage_options(input_uri)
    path = PurePosixPath(parts["path"])
    zarr_path = path.with_name(path.name + ".zarr")  # append .zarr to full filename
    return safe_uri_join(base_uri, str(zarr_path))
