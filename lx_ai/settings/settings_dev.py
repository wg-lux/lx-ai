import os

from .settings_base import *  # noqa: F403,F401

DEBUG = True
ALLOWED_HOSTS = ["*"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "dev_db.sqlite")),  # noqa: F405
    }
}
