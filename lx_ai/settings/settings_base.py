import os
from pathlib import Path
from lx_ai.settings.config import load_config

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "django-insecure-dev-only-change-me")

DEBUG = False
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    "lx_ai",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # required for endoreg DB models
    "endoreg_db",
    # lx-data-models Django app.
    # Python module path is lx_dtypes.django,
    # Django app label is lx_dtypes_django.
    "lx_dtypes.django.apps.LxDtypesDjangoConfig",
    # useful tools
    "django_extensions",
    "rest_framework",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

STATIC_URL = "/static/"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"

USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = REPO_ROOT

_settings_module = os.getenv("DJANGO_SETTINGS_MODULE", "")
_is_dev_settings = _settings_module.endswith("settings_dev")

"""if not _is_dev_settings:
    _env_path = BASE_DIR / ".env.systemd"
else:
    _env_path = BASE_DIR / ".env"

config = load_config(env_file=_env_path if _env_path.exists() else None)"""
config = load_config()

DEBUG = config.debug

DATABASES = {
    "ENGINE": "sqlite",
    "NAME": BASE_DIR / "db.sqlite3",
}

print("\n========== SETTINGS DEBUG ==========")
print(f"os.environ DATA_DIR: {os.getenv('DATA_DIR')}")
print(f"os.environ CONF_DIR: {os.getenv('CONF_DIR')}")
print(f"os.environ FRAME_DIR: {os.getenv('FRAME_DIR')}")
print("====================================\n")

DATA_DIR = config.data_dir
CONF_DIR = config.conf_dir

print("\n========== RESOLVED PATHS ==========")
print(f"CONFIG DATA_DIR: {DATA_DIR}")
print(f"CONFIG CONF_DIR: {CONF_DIR}")
print(f"CONFIG FRAME_DIR (before logic): {config.frame_dir}")
print("===================================\n")

# Development should always use local frames
if _is_dev_settings:
    FRAME_DIR = BASE_DIR / "data" / "frames"
else:
    FRAME_DIR = config.frame_dir
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONF_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
