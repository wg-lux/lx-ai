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

  SYNC_CMD = "uv sync --extra dev --extra docs";

in
{
  secretspec.provider = "env";

  # A dotenv file was found, while dotenv integration is currently not enabled.
  dotenv.enable = false;
  dotenv.disableHint = true;

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
    lxai_training.exec = "           
    source .devenv/state/venv/bin/activate
    secretspec run --provider env uv run python -m lx_ai.training";

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

    if [ -f ".env.systemd" ]; then
      set -a
      source .env.systemd
      set +a
      echo ".env.systemd file loaded successfully."
    else
      echo "Note: .env.systemd not found. Defaults apply."
    fi
  '';

  enterTest = ''
    nvcc -V
    pytest --maxfail=1 --disable-warnings -q
  '';
}
