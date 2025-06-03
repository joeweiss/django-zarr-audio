from huey import MemoryHuey

HUEY = MemoryHuey()


SECRET_KEY = "test"
DEBUG = True
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "huey.contrib.djhuey",
    # "bx_django_utils",
    # "huey_monitor",
    "django_zarr_audio",
]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
ROOT_URLCONF = "tests.urls"
