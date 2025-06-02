def test_import():
    import django_zarr_audio

    assert hasattr(django_zarr_audio, "__version__")
