import os
from pathlib import Path

from lx_ai.settings.config import load_config

from .settings_base import *  # noqa: F403,F401

config = load_config()

DEBUG = False


DATABASES = {
    "default": {
        "ENGINE": config.db_engine,
        "NAME": config.db_name,
        "USER": config.db_user,
        "PASSWORD": config.db_password,
        "HOST": config.db_host,
        "PORT": config.db_port,
        "OPTIONS": (
            {"sslmode": config.db_sslmode}
            if config.db_engine.endswith("postgresql")
            else {}
        ),
    }
}


# Django FileField / encrypted storage root.
#
# In production lx-ai reads protected media owned by lx-annotate.
# VideoFile.processed_file is stored as a relative path, e.g.
#   processed_videos_final/<hash>.mp4
#
# Therefore Django MEDIA_ROOT must point to the protected storage root,
# not to the lx-ai repository directory.
MEDIA_ROOT = (
    os.environ.get("MEDIA_ROOT")
    or os.environ.get("PROTECTED_MEDIA_ROOT")
    or os.environ.get("STORAGE_DIR")
    or globals().get("MEDIA_ROOT", "")
)

if MEDIA_ROOT:
    MEDIA_ROOT = str(Path(MEDIA_ROOT).resolve())
