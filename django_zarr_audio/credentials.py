import os
import fsspec
import warnings


def get_fs_from_env(label: str, backend: str):
    prefix = f"DJZA_{label}_"

    if backend == "s3":
        key = os.getenv(f"{prefix}KEY")
        secret = os.getenv(f"{prefix}SECRET")
        region = os.getenv(f"{prefix}REGION", "us-east-1")

        if key and secret:
            return fsspec.filesystem(
                "s3",
                key=key,
                secret=secret,
                client_kwargs={"region_name": region},
                use_ssl=True,
                skip_instance_cache=True,  # prevent caching a broken client
            )
        else:
            warnings.warn(f"Using environment's AWS credential chain for '{label}'")
            return fsspec.filesystem(
                "s3",
                skip_instance_cache=True,  # prevents reuse of broken/default clients
            )
    elif backend == "gcs":
        token_path = os.getenv(f"{prefix}TOKEN_PATH")
        if token_path and not os.path.exists(token_path):
            raise FileNotFoundError(f"GCS token file not found: {token_path}")
        return fsspec.filesystem("gcs", token=token_path)
    elif backend == "http":
        return fsspec.filesystem("http")
    elif backend == "file":
        return fsspec.filesystem("file")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
