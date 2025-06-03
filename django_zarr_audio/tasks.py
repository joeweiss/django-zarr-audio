from huey.contrib.djhuey import task, lock_task


@task()
@lock_task(lambda audio_file_id: f"zarr-encode-{audio_file_id}")
def run_zarr_encoding(audio_file_id):
    from .models import AudioFile
    from zarr_audio.encoder import AudioEncoder
    from .utils import get_storage_options

    audio_file = AudioFile.objects.get(id=audio_file_id)

    if audio_file.status != AudioFile.STATUS.queued:
        return

    try:
        audio_file.status = AudioFile.STATUS.encoding
        audio_file.save()

        output_profile = audio_file.storage_mapping.output_profile
        storage_options = get_storage_options(output_profile.credentials_label)
        zarr_uri = audio_file.zarr_uri

        encoder = AudioEncoder(
            input_uri=audio_file.uri,
            output_uri=zarr_uri,
            storage_options=storage_options,
            chunk_duration=10,
        )
        encoder.encode()

        audio_file.status = AudioFile.STATUS.encoded
    except Exception as e:
        audio_file.status = AudioFile.STATUS.exception_returned

    audio_file.save()
