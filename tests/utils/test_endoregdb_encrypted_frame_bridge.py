# tests/utils/test_endoregdb_encrypted_frame_bridge.py

from pathlib import Path

from endoreg_db.utils.encryption.encryption import MAGIC
from lx_ai.utils.endoregdb_encrypted_frame_bridge import _path_is_encrypted


def test_path_is_encrypted_detects_magic(tmp_path: Path):
    encrypted = tmp_path / "video.mp4"
    encrypted.write_bytes(MAGIC + b"ciphertext")

    assert _path_is_encrypted(encrypted) is True


def test_path_is_encrypted_false_for_plain_mp4(tmp_path: Path):
    plain = tmp_path / "video.mp4"
    plain.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    assert _path_is_encrypted(plain) is False


def test_path_is_encrypted_false_for_missing_file(tmp_path: Path):
    assert _path_is_encrypted(tmp_path / "missing.mp4") is False
