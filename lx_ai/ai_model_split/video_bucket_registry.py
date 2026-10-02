# lx_ai/ai_model_split/video_bucket_registry.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, TypedDict


class VideoBucketRegistryData(TypedDict):
    version: int
    num_buckets: int
    videos: Dict[str, int]


class VideoBucketRegistry:
    """
    Persistent registry of stable video_key -> bucket_id assignments.

    Guarantees:
      - existing videos keep the same bucket forever
      - assignments survive reruns / new dataset arrivals
      - atomic save to avoid partial writes
    """

    def __init__(self, path: Path, num_buckets: int) -> None:
        self.path = path
        self.num_buckets = int(num_buckets)
        self.version = 1
        self.videos: Dict[str, int] = {}

    @classmethod
    def load(cls, path: Path, num_buckets: int) -> "VideoBucketRegistry":
        reg = cls(path=path, num_buckets=num_buckets)
        if not path.exists():
            return reg

        raw = json.loads(path.read_text(encoding="utf-8"))
        stored_num_buckets = int(raw["num_buckets"])
        if stored_num_buckets != int(num_buckets):
            raise ValueError(
                f"Bucket registry at {path} was created with num_buckets={stored_num_buckets}, "
                f"but current policy uses num_buckets={num_buckets}. Refusing to continue."
            )

        reg.version = int(raw.get("version", 1))
        reg.videos = {str(k): int(v) for k, v in raw.get("videos", {}).items()}
        return reg

    def get(self, video_key: str) -> int | None:
        return self.videos.get(video_key)

    def set(self, video_key: str, bucket_id: int) -> None:
        bucket_id = int(bucket_id)
        if not (0 <= bucket_id < self.num_buckets):
            raise ValueError(
                f"bucket_id out of range: {bucket_id} for num_buckets={self.num_buckets}"
            )
        self.videos[video_key] = bucket_id

    def to_dict(self) -> VideoBucketRegistryData:
        return {
            "version": int(self.version),
            "num_buckets": int(self.num_buckets),
            "videos": dict(sorted(self.videos.items())),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
