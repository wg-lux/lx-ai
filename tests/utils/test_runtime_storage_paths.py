# tests/utils/test_runtime_storage_paths.py

from pathlib import Path


def test_lxai_storage_env_contract(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    storage_dir = data_dir / "storage"

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DJANGO_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LX_ANNOTATE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LX_ANNOTATE_ENCRYPTED_DATA_DIR", str(data_dir))
    monkeypatch.setenv("STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("PROTECTED_MEDIA_ROOT", str(storage_dir))

    processed_name = "processed_videos_final/example.mp4"
    expected = storage_dir / processed_name

    assert expected == Path(
        Path(__import__("os").environ["PROTECTED_MEDIA_ROOT"]) / processed_name
    )
