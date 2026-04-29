{
  pkgs,
  lib,
  config,
  inputs,
  baseBuildInputs,
  ...
}:
let
  # Pin to specific Python 3.12 version to match pyproject.toml
  python = pkgs.python312; # known devenv issue with python3Packages since python3Full was deprecated
  uvPackage = pkgs.uv;

  buildInputs = with pkgs; [
    python312
    stdenv.cc.cc
    tesseract
    glib
    openssh
    cmake
    gcc
    pkg-config
    protobuf
    libglvnd
  ];
  runtimePackages = with pkgs; [
    stdenv.cc.cc
    ffmpeg-headless.bin
    tesseract
    uvPackage
    libglvnd
    glib
    zlib
    secretspec
    xorg.libxcb
  ];

  baseEnv = {
    FRAME_DIR = config.secretspec.secrets.FRAME_DIR;

    TRAINING_CONFIG_PATH = config.secretspec.secrets.TRAINING_CONFIG_PATH;

    TRAINING_ROOT = config.secretspec.secrets.TRAINING_ROOT;
    CHECKPOINTS_DIR = config.secretspec.secrets.CHECKPOINTS_DIR;
    RUNS_DIR = config.secretspec.secrets.RUNS_DIR;
    BUCKET_SNAPSHOT_DIR = config.secretspec.secrets.BUCKET_SNAPSHOT_DIR;

    BACKBONE_CHECKPOINT = config.secretspec.secrets.BACKBONE_CHECKPOINT;
    BACKBONE_CHECKPOINT_URL = config.secretspec.secrets.BACKBONE_CHECKPOINT_URL;
    SQLITE_DB_PATH = config.secretspec.secrets.SQLITE_DB_PATH;

    LEGACY_IMAGE_DIR = config.secretspec.secrets.LEGACY_IMAGE_DIR;
    LEGACY_JSONL_PATH = config.secretspec.secrets.LEGACY_JSONL_PATH;

    CSV_DIR = config.secretspec.secrets.CSV_DIR;

    FRAME_PATH_REMAP_SOURCE = config.secretspec.secrets.FRAME_PATH_REMAP_SOURCE;
    FRAME_PATH_REMAP_TARGET = config.secretspec.secrets.FRAME_PATH_REMAP_TARGET;

    HOME_DIR = config.secretspec.secrets.HOME_DIR;
    WORKING_DIR = config.secretspec.secrets.WORKING_DIR;
    DATA_DIR = config.secretspec.secrets.DATA_DIR;
    CONF_DIR = config.secretspec.secrets.CONF_DIR;

    STORAGE_DIR = config.secretspec.secrets.DATA_DIR;

    DJANGO_ENV = config.secretspec.secrets.DJANGO_ENV;
    DJANGO_DEBUG = config.secretspec.secrets.DJANGO_DEBUG;

    DJANGO_SETTINGS_MODULE = config.secretspec.secrets.DJANGO_SETTINGS_MODULE;
    DJANGO_SETTINGS_MODULE_DEVELOPMENT = config.secretspec.secrets.DJANGO_SETTINGS_MODULE_DEVELOPMENT;
    DJANGO_SETTINGS_MODULE_PRODUCTION = config.secretspec.secrets.DJANGO_SETTINGS_MODULE_PRODUCTION;

    DJANGO_DB_ENGINE = config.secretspec.secrets.DJANGO_DB_ENGINE;
    DJANGO_DB_HOST = config.secretspec.secrets.DJANGO_DB_HOST;
    DJANGO_DB_PORT = config.secretspec.secrets.DJANGO_DB_PORT;
    DJANGO_DB_NAME = config.secretspec.secrets.DJANGO_DB_NAME;
    DJANGO_DB_USER = config.secretspec.secrets.DJANGO_DB_USER;
    DJANGO_DB_PASSWORD_FILE = config.secretspec.secrets.DJANGO_DB_PASSWORD_FILE;
    DJANGO_DB_SSLMODE = config.secretspec.secrets.DJANGO_DB_SSLMODE;

    LOG_LEVEL = config.secretspec.secrets.LOG_LEVEL;
};

  _module.args.buildInputs = baseBuildInputs;

  SYNC_CMD = "uv sync --active --extra dev --extra docs";

in
{
  secretspec.provider = "env";

  # A dotenv file was found, while dotenv integration is currently not enabled.
  dotenv.enable = false;
  dotenv.disableHint = true;
  cachix.enable = false;
  packages = runtimePackages ++ buildInputs;

  env = baseEnv // {

    # include runtimePackages as well so runtime native libs (e.g. zlib) are on LD_LIBRARY_PATH
    LD_LIBRARY_PATH =
      lib.makeLibraryPath (buildInputs ++ runtimePackages)
      + ":/run/opengl-driver/lib:/run/opengl-driver-32/lib";

  };

  languages.python = {
    enable = true;
    package = python;
    uv = {
      enable = true;
      package = uvPackage;
      sync.enable = true;
    };
  };

  scripts = {
    env-setup.exec = ''
      # Ensure runtimePackages are included in the library path here too
      export LD_LIBRARY_PATH="${
        with pkgs; lib.makeLibraryPath (buildInputs ++ runtimePackages)
      }:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
    '';
    prepare-assets.exec = ''
      source .devenv/state/venv/bin/activate
      secretspec run --provider env -- uv run python lx_ai/scripts/prepare_runtime_assets.py
    '';

    lxai_training.exec = ''
      source .devenv/state/venv/bin/activate
      secretspec run --provider env -- uv run python lx_ai/scripts/prepare_runtime_assets.py
      secretspec run --provider env -- uv run python lx_ai/run_training.py
    '';

    pyshell.exec = "uv run python manage.py shell";

    mkdocs.exec = ''
      uv run make -C docs html
      uv run make -C docs linkcheck
    '';
    uvsnc.exec = ''
      ${SYNC_CMD}
    '';
  };

  tasks = {
  };

  processes = {
  };

  enterShell = ''
    export SYNC_CMD="${SYNC_CMD}"

    # Ensure dependencies are synced using uv
    # Check if venv exists. If not, run sync verbosely. If it exists, sync quietly.
    if [ ! -d ".devenv/state/venv" ]; then
       echo "Virtual environment not found. Running initial uv sync..."
       $SYNC_CMD || echo "Error: Initial uv sync failed. Please check network and pyproject.toml."
    else
       # Sync quietly if venv exists
       echo "Syncing Python dependencies with uv..."
       $SYNC_CMD --quiet || echo "Warning: uv sync failed. Environment might be outdated."
    fi
    env-setup

    if [ -f ".env" ]; then
      set -a
      source .env
      set +a

      export DJANGO_ENV=''${DJANGO_ENV:-development}
      export DATA_DIR=''${DATA_DIR:-data}
      export CONF_DIR=''${CONF_DIR:-conf}
      export FRAME_DIR=''${FRAME_DIR:-data/frames}
      export STORAGE_DIR=''${STORAGE_DIR:-data}

      export TRAINING_CONFIG_PATH=''${TRAINING_CONFIG_PATH:-lx_ai/ai_model_config/train_sandbox_postgres.yaml}

      export TRAINING_ROOT=''${TRAINING_ROOT:-data/model_training}
      export CHECKPOINTS_DIR=''${CHECKPOINTS_DIR:-data/model_training/checkpoints}
      export RUNS_DIR=''${RUNS_DIR:-data/model_training/runs}
      export BUCKET_SNAPSHOT_DIR=''${BUCKET_SNAPSHOT_DIR:-data/model_training/buckets}

      export BACKBONE_CHECKPOINT=''${BACKBONE_CHECKPOINT:-data/model_training/checkpoints/RN50_GastroNet-1M_DINOv1.pth}
      export BACKBONE_CHECKPOINT_URL=''${BACKBONE_CHECKPOINT_URL:-}

      export SQLITE_DB_PATH=''${SQLITE_DB_PATH:-dev_db.sqlite}

      export LEGACY_IMAGE_DIR=''${LEGACY_IMAGE_DIR:-data/legacy_images/images}
      export LEGACY_JSONL_PATH=''${LEGACY_JSONL_PATH:-data/legacy_images/legacy_img_dicts.jsonl}

      export CSV_DIR=''${CSV_DIR:-data/import/csv}

      export FRAME_PATH_REMAP_SOURCE=''${FRAME_PATH_REMAP_SOURCE:-}
      export FRAME_PATH_REMAP_TARGET=''${FRAME_PATH_REMAP_TARGET:-}

      echo ".env (dev) loaded"
    elif [ -f ".env.systemd" ]; then
      set -a
      source .env.systemd
      set +a
      echo ".env.systemd (fallback) loaded"
    fi
    '';

  enterTest = ''
    nvcc -V
    pytest --maxfail=1 --disable-warnings -q
  '';
}
