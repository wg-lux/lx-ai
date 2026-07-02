from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_split.video_bucket_allocator import (
    assign_buckets_with_persistent_video_registry,
)
from lx_ai.ai_model_split.video_bucket_registry import VideoBucketRegistry


class TestVideoBucketAllocator:
    def _config(
        self,
        tmp_path: Path,
        *,
        treat_unlabeled_as_negative: bool = False,
        num_buckets: int = 5,
        validation_bucket: int = 3,
        test_bucket: int = 4,
    ) -> TrainingConfig:
        # creates a valid TrainingConfig for allocator tests
        return TrainingConfig.model_validate(
            {
                "dataset_uuid": "test_dataset",
                "data_source": "postgres",
                "dataset_ids": [1],
                "labelset_id": 5,
                "labelset_version_to_train": 3,
                "treat_unlabeled_as_negative": treat_unlabeled_as_negative,
                "base_dir": str(tmp_path),
                "training_root": str(tmp_path / "training"),
                "checkpoints_dir": str(tmp_path / "training" / "checkpoints"),
                "runs_dir": str(tmp_path / "training" / "runs"),
                "create_dirs": True,
                "backbone_name": "gastro_rn50",
                "backbone_checkpoint": None,
                "freeze_backbone": True,
                "num_epochs": 1,
                "batch_size": 2,
                "lr_head": 0.001,
                "lr_backbone": 0.0001,
                "gamma_focal": 2.0,
                "alpha_focal": 0.25,
                "use_scheduler": False,
                "warmup_epochs": 0,
                "min_lr": 1.0e-6,
                "device": "cpu",
                "random_seed": 42,
                "bucket_policy": {
                    "num_buckets": num_buckets,
                    "validation_buckets": [validation_bucket],
                    "test_buckets": [test_bucket],
                },
                "save_bucket_snapshot": False,
            }
        )

    def _basic_inputs(self) -> dict[str, Any]:
        # creates simple valid video based training metadata
        return {
            "video_ids": [100, 100, 200, 200, 300, 300],
            "dataset_ids_per_frame": [1, 1, 1, 1, 1, 1],
            "label_vectors": [
                [1, None],
                [1, None],
                [None, 1],
                [None, 1],
                [1, None],
                [None, 1],
            ],
            "label_masks": [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            "label_names": ["polyp", "blood"],
        }

    def _partial_with_negative_inputs(self) -> dict[str, Any]:
        # creates input where at least one true negative exists
        return {
            "video_ids": [100, 100, 200, 200],
            "dataset_ids_per_frame": [1, 1, 1, 1],
            "label_vectors": [
                [1, 0],
                [1, None],
                [0, 1],
                [None, 1],
            ],
            "label_masks": [
                [1, 1],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            "label_names": ["polyp", "blood"],
        }

    def test_allocator_returns_expected_result_keys(self, tmp_path: Path) -> None:
        # checks allocator returns all required output keys
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert set(result.keys()) == {
            "bucket_ids_per_sample",
            "bucket_sizes",
            "role_sizes",
            "train_indices",
            "val_indices",
            "test_indices",
            "bucket_map",
            "diagnostics",
        }

    def test_allocator_assigns_one_bucket_per_sample(self, tmp_path: Path) -> None:
        # checks each input frame receives exactly one bucket id
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert len(result["bucket_ids_per_sample"]) == len(inputs["video_ids"])

    def test_allocator_bucket_ids_are_inside_valid_range(self, tmp_path: Path) -> None:
        # checks all produced bucket ids are inside configured bucket range
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        for bucket_id in result["bucket_ids_per_sample"]:
            assert 0 <= bucket_id < config.bucket_policy.num_buckets

    def test_allocator_keeps_same_video_in_same_bucket(self, tmp_path: Path) -> None:
        # checks all frames from same video get same bucket
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        video_to_bucket: dict[int, int] = {}

        for video_id, bucket_id in zip(
            inputs["video_ids"],
            result["bucket_ids_per_sample"],
        ):
            if video_id not in video_to_bucket:
                video_to_bucket[video_id] = bucket_id
            else:
                assert video_to_bucket[video_id] == bucket_id

    def test_allocator_split_indices_cover_all_samples_once(
        self, tmp_path: Path
    ) -> None:
        # checks train validation and test indices cover all samples exactly once
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        all_indices = (
            result["train_indices"] + result["val_indices"] + result["test_indices"]
        )

        assert sorted(all_indices) == list(range(len(inputs["video_ids"])))
        assert len(all_indices) == len(set(all_indices))

    def test_allocator_split_indices_match_bucket_policy(self, tmp_path: Path) -> None:
        # checks split role is created from bucket policy
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        val_buckets = set(config.bucket_policy.validation_buckets)
        test_buckets = set(config.bucket_policy.test_buckets)

        for idx in result["val_indices"]:
            assert result["bucket_ids_per_sample"][idx] in val_buckets

        for idx in result["test_indices"]:
            assert result["bucket_ids_per_sample"][idx] in test_buckets

        for idx in result["train_indices"]:
            bucket_id = result["bucket_ids_per_sample"][idx]
            assert bucket_id not in val_buckets
            assert bucket_id not in test_buckets

    def test_allocator_role_sizes_match_split_lengths(self, tmp_path: Path) -> None:
        # checks role_sizes match actual train validation and test lengths
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert result["role_sizes"]["train"] == len(result["train_indices"])
        assert result["role_sizes"]["val"] == len(result["val_indices"])
        assert result["role_sizes"]["test"] == len(result["test_indices"])

    def test_allocator_bucket_sizes_match_bucket_ids(self, tmp_path: Path) -> None:
        # checks bucket_sizes counts all produced sample bucket ids
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert sum(result["bucket_sizes"].values()) == len(
            result["bucket_ids_per_sample"]
        )

    def test_allocator_is_deterministic_with_existing_registry(
        self, tmp_path: Path
    ) -> None:
        # checks second run with same registry gives same result
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result_1 = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )
        result_2 = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert result_1["bucket_ids_per_sample"] == result_2["bucket_ids_per_sample"]
        assert result_1["bucket_map"] == result_2["bucket_map"]
        assert result_1["train_indices"] == result_2["train_indices"]
        assert result_1["val_indices"] == result_2["val_indices"]
        assert result_1["test_indices"] == result_2["test_indices"]

    def test_allocator_creates_registry_file(self, tmp_path: Path) -> None:
        # checks allocator saves video bucket registry on disk
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        registry_path = (
            Path(config.training_root)
            / "bucket_registry"
            / "video_bucket_registry.json"
        )

        assert registry_path.exists()

    def test_allocator_saved_registry_contains_all_videos(self, tmp_path: Path) -> None:
        # checks registry contains one entry per unique video
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        registry_path = (
            Path(config.training_root)
            / "bucket_registry"
            / "video_bucket_registry.json"
        )
        registry = VideoBucketRegistry.load(
            path=registry_path,
            num_buckets=config.bucket_policy.num_buckets,
        )

        expected_video_keys = {f"video:{v}" for v in set(inputs["video_ids"])}

        assert set(registry.videos.keys()) == expected_video_keys

    def test_allocator_reuses_existing_registry_assignment(
        self, tmp_path: Path
    ) -> None:
        # checks existing video keeps its old bucket assignment
        config = self._config(tmp_path)
        registry_path = (
            Path(config.training_root)
            / "bucket_registry"
            / "video_bucket_registry.json"
        )

        registry = VideoBucketRegistry.load(
            path=registry_path,
            num_buckets=config.bucket_policy.num_buckets,
        )
        registry.set("video:100", 3)
        registry.save()

        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert result["bucket_map"]["video:100"] == 3

        for video_id, bucket_id in zip(
            inputs["video_ids"],
            result["bucket_ids_per_sample"],
        ):
            if video_id == 100:
                assert bucket_id == 3

    def test_allocator_adds_new_video_without_changing_existing_video(
        self,
        tmp_path: Path,
    ) -> None:
        # checks new videos are added and old video bucket remains unchanged
        config = self._config(tmp_path)

        first_inputs = {
            "video_ids": [100, 100],
            "dataset_ids_per_frame": [1, 1],
            "label_vectors": [[1, None], [1, None]],
            "label_masks": [[1, 0], [1, 0]],
            "label_names": ["polyp", "blood"],
        }

        first_result = assign_buckets_with_persistent_video_registry(
            config=config,
            **first_inputs,
        )
        old_bucket = first_result["bucket_map"]["video:100"]

        second_inputs = {
            "video_ids": [100, 100, 200, 200],
            "dataset_ids_per_frame": [1, 1, 1, 1],
            "label_vectors": [[1, None], [1, None], [None, 1], [None, 1]],
            "label_masks": [[1, 0], [1, 0], [0, 1], [0, 1]],
            "label_names": ["polyp", "blood"],
        }

        second_result = assign_buckets_with_persistent_video_registry(
            config=config,
            **second_inputs,
        )

        assert second_result["bucket_map"]["video:100"] == old_bucket
        assert "video:200" in second_result["bucket_map"]

    def test_allocator_raises_when_registry_num_buckets_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        # checks allocator refuses registry created with different num_buckets
        config = self._config(tmp_path)

        registry_path = (
            Path(config.training_root)
            / "bucket_registry"
            / "video_bucket_registry.json"
        )

        registry = VideoBucketRegistry.load(path=registry_path, num_buckets=4)
        registry.set("video:100", 1)
        registry.save()

        inputs = self._basic_inputs()

        with pytest.raises(ValueError, match="was created with num_buckets=4"):
            assign_buckets_with_persistent_video_registry(
                config=config,
                **inputs,
            )

    def test_allocator_detects_positives_only_condition(self, tmp_path: Path) -> None:
        # checks positives only mode is detected when no known negatives exist
        config = self._config(tmp_path, treat_unlabeled_as_negative=False)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert result["diagnostics"]["condition"] == "positives_only"

    def test_allocator_detects_partial_with_negatives_condition(
        self,
        tmp_path: Path,
    ) -> None:
        # checks partial mode with true negatives is detected
        config = self._config(tmp_path, treat_unlabeled_as_negative=False)
        inputs = self._partial_with_negative_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert result["diagnostics"]["condition"] == "partial_with_negatives"

    def test_allocator_detects_closed_world_condition(self, tmp_path: Path) -> None:
        # checks closed world mode is detected when unlabeled labels are negatives
        config = self._config(tmp_path, treat_unlabeled_as_negative=True)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert result["diagnostics"]["condition"] == "closed_world"

    def test_allocator_diagnostics_contains_video_grouping(
        self, tmp_path: Path
    ) -> None:
        # checks diagnostics include video grouping summary
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        grouping = result["diagnostics"]["video_grouping"]

        assert grouping["total_videos"] == 3
        assert grouping["total_frames"] == 6
        assert grouping["total_datasets"] == 1

    def test_allocator_diagnostics_contains_final_assignments(
        self,
        tmp_path: Path,
    ) -> None:
        # checks diagnostics include final video bucket assignments
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        final_assignments = result["diagnostics"]["final_assignments"]

        assert len(final_assignments) == 3
        assert {row["video_key"] for row in final_assignments} == {
            "video:100",
            "video:200",
            "video:300",
        }

    def test_allocator_diagnostics_contains_bucket_balance(
        self,
        tmp_path: Path,
    ) -> None:
        # checks diagnostics include one bucket balance row per bucket
        config = self._config(tmp_path)
        inputs = self._basic_inputs()

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        bucket_balance = result["diagnostics"]["bucket_balance"]

        assert len(bucket_balance) == config.bucket_policy.num_buckets

    def test_allocator_works_with_single_video(self, tmp_path: Path) -> None:
        # checks allocator works when there is only one video
        config = self._config(tmp_path)

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            video_ids=[100, 100],
            dataset_ids_per_frame=[1, 1],
            label_vectors=[[1, None], [1, None]],
            label_masks=[[1, 0], [1, 0]],
            label_names=["polyp", "blood"],
        )

        assert len(result["bucket_ids_per_sample"]) == 2
        assert len(set(result["bucket_ids_per_sample"])) == 1
        assert list(result["bucket_map"].keys()) == ["video:100"]

    def test_allocator_uses_multiple_buckets_for_multiple_videos(
        self,
        tmp_path: Path,
    ) -> None:
        # checks allocator does not put many videos into only one bucket
        config = self._config(tmp_path)

        video_ids = []
        dataset_ids_per_frame = []
        label_vectors = []
        label_masks = []

        for video_id in range(100, 110):
            video_ids.extend([video_id, video_id])
            dataset_ids_per_frame.extend([1, 1])
            label_vectors.extend([[1, None], [None, 1]])
            label_masks.extend([[1, 0], [0, 1]])

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            video_ids=video_ids,
            dataset_ids_per_frame=dataset_ids_per_frame,
            label_vectors=label_vectors,
            label_masks=label_masks,
            label_names=["polyp", "blood"],
        )

        assert len(set(result["bucket_map"].values())) > 1

    def test_allocator_raises_on_empty_input(self, tmp_path: Path) -> None:
        # checks allocator rejects empty input
        config = self._config(tmp_path)

        with pytest.raises(ValueError, match="No samples to assign"):
            assign_buckets_with_persistent_video_registry(
                config=config,
                video_ids=[],
                dataset_ids_per_frame=[],
                label_vectors=[],
                label_masks=[],
                label_names=["polyp", "blood"],
            )

    def test_allocator_raises_when_video_ids_do_not_align(
        self,
        tmp_path: Path,
    ) -> None:
        # checks video_ids must align with label vectors and masks
        config = self._config(tmp_path)

        with pytest.raises(ValueError, match="must align"):
            assign_buckets_with_persistent_video_registry(
                config=config,
                video_ids=[100],
                dataset_ids_per_frame=[1, 1],
                label_vectors=[[1, None], [None, 1]],
                label_masks=[[1, 0], [0, 1]],
                label_names=["polyp", "blood"],
            )

    def test_allocator_raises_when_dataset_ids_do_not_align(
        self,
        tmp_path: Path,
    ) -> None:
        # checks dataset ids must align with all frames
        config = self._config(tmp_path)

        with pytest.raises(ValueError, match="must align"):
            assign_buckets_with_persistent_video_registry(
                config=config,
                video_ids=[100, 100],
                dataset_ids_per_frame=[1],
                label_vectors=[[1, None], [None, 1]],
                label_masks=[[1, 0], [0, 1]],
                label_names=["polyp", "blood"],
            )

    def test_allocator_raises_when_label_masks_do_not_align(
        self,
        tmp_path: Path,
    ) -> None:
        # checks label masks must align with label vectors
        config = self._config(tmp_path)

        with pytest.raises(ValueError, match="must align"):
            assign_buckets_with_persistent_video_registry(
                config=config,
                video_ids=[100, 100],
                dataset_ids_per_frame=[1, 1],
                label_vectors=[[1, None], [None, 1]],
                label_masks=[[1, 0]],
                label_names=["polyp", "blood"],
            )

    def test_allocator_accepts_missing_label_names(self, tmp_path: Path) -> None:
        # checks label_names can be omitted
        config = self._config(tmp_path)
        inputs = self._basic_inputs()
        inputs.pop("label_names")

        result = assign_buckets_with_persistent_video_registry(
            config=config,
            **inputs,
        )

        assert len(result["bucket_ids_per_sample"]) == len(inputs["video_ids"])
