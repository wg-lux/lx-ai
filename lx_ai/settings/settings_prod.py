from .settings_base import *  # noqa: F403,F401
from lx_ai.settings.config import load_config

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
    }
}
