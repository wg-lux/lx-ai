# lx_ai/scripts/prepare_runtime_assets.py
from __future__ import annotations

import os
import urllib.request
from pathlib import Path


REQUIRED_ASSETS = [
    {
        "name": "GastroNet RN50 checkpoint",
        "env_path": "BACKBONE_CHECKPOINT",
        "url_env": "BACKBONE_CHECKPOINT_URL",
        "required": True,
    },
]

# TODO commenting it as currently , prvt link to download checkpoint provided, need to update with actual url in .env and luxnix lx-ai-local
"""def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")

    print(f"[DOWNLOAD] {url}")
    print(f"[TARGET]   {target}")

    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)"""


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")

    print(f"[DOWNLOAD] {url}")
    print(f"[TARGET]   {target}")

    if "drive.google.com" in url:
        import gdown

        # gdown.download(url=url, output=str(tmp), quiet=False, fuzzy=True)
        gdown.download(url, str(tmp), quiet=False)
    else:
        urllib.request.urlretrieve(url, tmp)

    if not tmp.is_file() or tmp.stat().st_size == 0:
        raise RuntimeError(f"Download failed or produced empty file: {tmp}")

    tmp.replace(target)


def main() -> None:
    missing_without_url: list[str] = []

    for asset in REQUIRED_ASSETS:
        target_raw = os.getenv(asset["env_path"], "")
        url = os.getenv(asset["url_env"], "")

        if not target_raw:
            missing_without_url.append(
                f"{asset['name']}: {asset['env_path']} is not set"
            )
            continue

        target = Path(target_raw).expanduser()

        if target.is_file():
            print(f"[OK] {asset['name']}: {target}")
            continue

        if not url:
            missing_without_url.append(
                f"{asset['name']}: missing {target}; set {asset['url_env']} to download it"
            )
            continue

        _download(url, target)

    if missing_without_url:
        print("\nMissing required runtime assets:")
        for item in missing_without_url:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nAll required runtime assets are present.")


if __name__ == "__main__":
    main()
