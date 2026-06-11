from __future__ import annotations

from pathlib import Path


def test_settings_prod_media_root_uses_protected_storage_fallbacks() -> None:
    """
    Production contract:

    Django MEDIA_ROOT must resolve protected media below lx-annotate storage,
    not below the lx-ai repository.

    This prevents VideoFile.processed_file from resolving to:
      /var/endoreg-service-user/lx-ai/processed_videos_final/...

    Expected fallback order:
      MEDIA_ROOT
      PROTECTED_MEDIA_ROOT
      STORAGE_DIR
    """
    settings_file = Path("lx_ai/settings/settings_prod.py")
    assert settings_file.exists(), "settings_prod.py must exist"

    source = settings_file.read_text(encoding="utf-8")

    assert "MEDIA_ROOT" in source
    assert 'os.environ.get("MEDIA_ROOT")' in source
    assert 'os.environ.get("PROTECTED_MEDIA_ROOT")' in source
    assert 'os.environ.get("STORAGE_DIR")' in source
    assert "Path(MEDIA_ROOT).resolve()" in source
