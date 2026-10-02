from __future__ import annotations

from pathlib import Path

import pytest

from lx_ai.utils.data_loader_for_model_training import build_image_multilabel_dataset


class TestDataLoaderForModelTraining:
    def _touch_image(self, path: Path) -> Path:
        # creates a fake image file because builder only checks file exists
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-image")
        return path

    def _labelset(self) -> dict:
        # creates labelset with ordered labels
        return {
            "id": 5,
            "version": 3,
            "labels": [
                {"id": 1, "name": "polyp"},
                {"id": 2, "name": "blood"},
            ],
        }

    def _annotations(self, tmp_path: Path) -> list[dict]:
        # creates annotation rows similar to database loader output
        self._touch_image(tmp_path / "frame_1.jpg")
        self._touch_image(tmp_path / "frame_2.jpg")

        return [
            {
                "dataset_id": 1,
                "frame": {
                    "id": 10,
                    "relative_path": "frame_1.jpg",
                    "file_path": str(tmp_path),
                    "old_examination_id": 100,
                    "video_id": 1000,
                },
                "label": {"id": 1, "name": "polyp"},
                "value": True,
                "annotator": "ann_a",
            },
            {
                "dataset_id": 1,
                "frame": {
                    "id": 11,
                    "relative_path": "frame_2.jpg",
                    "file_path": str(tmp_path),
                    "old_examination_id": 200,
                    "video_id": 2000,
                },
                "label": {"id": 2, "name": "blood"},
                "value": True,
                "annotator": "ann_b",
            },
        ]

    def test_build_image_multilabel_dataset_closed_world_vectors(
        self,
        tmp_path: Path,
    ) -> None:
        # checks missing labels become negative when treat_unlabeled_as_negative is true
        ds = build_image_multilabel_dataset(
            dataset_uuid="test_ds",
            annotations=self._annotations(tmp_path),
            labelset=self._labelset(),
            treat_unlabeled_as_negative=True,
        )

        assert ds["label_vectors"] == [
            [1, 0],
            [0, 1],
        ]
        assert ds["label_masks"] == [
            [1, 1],
            [1, 1],
        ]

    def test_build_image_multilabel_dataset_open_world_vectors(
        self,
        tmp_path: Path,
    ) -> None:
        # checks missing labels stay unknown when treat_unlabeled_as_negative is false
        ds = build_image_multilabel_dataset(
            dataset_uuid="test_ds",
            annotations=self._annotations(tmp_path),
            labelset=self._labelset(),
            treat_unlabeled_as_negative=False,
        )

        assert ds["label_vectors"] == [
            [1, None],
            [None, 1],
        ]
        assert ds["label_masks"] == [
            [1, 0],
            [0, 1],
        ]

    def test_build_image_multilabel_dataset_preserves_label_order(
        self,
        tmp_path: Path,
    ) -> None:
        # checks label columns follow labelset labels order
        ds = build_image_multilabel_dataset(
            dataset_uuid="test_ds",
            annotations=self._annotations(tmp_path),
            labelset=self._labelset(),
            treat_unlabeled_as_negative=True,
        )

        assert [label["name"] for label in ds["labels"]] == ["polyp", "blood"]

    def test_build_image_multilabel_dataset_collects_frame_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        # checks frame ids exam ids dataset ids video ids and annotators are returned
        ds = build_image_multilabel_dataset(
            dataset_uuid="test_ds",
            annotations=self._annotations(tmp_path),
            labelset=self._labelset(),
            treat_unlabeled_as_negative=True,
        )

        assert ds["frame_ids"] == [10, 11]
        assert ds["old_examination_ids"] == [100, 200]
        assert ds["dataset_ids_per_frame"] == [1, 1]
        assert ds["video_ids"] == [1000, 2000]
        assert ds["annotators_per_frame"] == [["ann_a"], ["ann_b"]]

    def test_build_image_multilabel_dataset_image_paths_are_resolved(
        self,
        tmp_path: Path,
    ) -> None:
        # checks image paths are returned as resolved strings
        ds = build_image_multilabel_dataset(
            dataset_uuid="test_ds",
            annotations=self._annotations(tmp_path),
            labelset=self._labelset(),
            treat_unlabeled_as_negative=True,
        )

        assert ds["image_paths"] == [
            str((tmp_path / "frame_1.jpg").resolve()),
            str((tmp_path / "frame_2.jpg").resolve()),
        ]

    def test_build_image_multilabel_dataset_groups_multiple_annotations_per_frame(
        self,
        tmp_path: Path,
    ) -> None:
        # checks multiple annotations on same frame produce one sample with two labels
        self._touch_image(tmp_path / "frame_1.jpg")

        annotations = [
            {
                "dataset_id": 1,
                "frame": {
                    "id": 10,
                    "relative_path": "frame_1.jpg",
                    "file_path": str(tmp_path),
                    "old_examination_id": 100,
                    "video_id": 1000,
                },
                "label": {"id": 1, "name": "polyp"},
                "value": True,
                "annotator": "ann_a",
            },
            {
                "dataset_id": 1,
                "frame": {
                    "id": 10,
                    "relative_path": "frame_1.jpg",
                    "file_path": str(tmp_path),
                    "old_examination_id": 100,
                    "video_id": 1000,
                },
                "label": {"id": 2, "name": "blood"},
                "value": True,
                "annotator": "ann_b",
            },
        ]

        ds = build_image_multilabel_dataset(
            dataset_uuid="test_ds",
            annotations=annotations,
            labelset=self._labelset(),
            treat_unlabeled_as_negative=True,
        )

        assert len(ds["image_paths"]) == 1
        assert ds["label_vectors"] == [[1, 1]]
        assert ds["label_masks"] == [[1, 1]]
        assert ds["annotators_per_frame"] == [["ann_a", "ann_b"]]

    def test_build_image_multilabel_dataset_drops_labels_with_zero_positive_samples(
        self,
        tmp_path: Path,
    ) -> None:
        # checks labels with no positive samples are removed by current builder logic
        self._touch_image(tmp_path / "frame_1.jpg")
        self._touch_image(tmp_path / "frame_2.jpg")

        annotations = [
            {
                "dataset_id": 1,
                "frame": {
                    "id": 10,
                    "relative_path": "frame_1.jpg",
                    "file_path": str(tmp_path),
                    "old_examination_id": 100,
                    "video_id": 1000,
                },
                "label": {"id": 1, "name": "polyp"},
                "value": False,
                "annotator": "ann_a",
            },
            {
                "dataset_id": 1,
                "frame": {
                    "id": 11,
                    "relative_path": "frame_2.jpg",
                    "file_path": str(tmp_path),
                    "old_examination_id": 200,
                    "video_id": 2000,
                },
                "label": {"id": 2, "name": "blood"},
                "value": True,
                "annotator": "ann_b",
            },
        ]

        ds = build_image_multilabel_dataset(
            dataset_uuid="test_ds",
            annotations=annotations,
            labelset=self._labelset(),
            treat_unlabeled_as_negative=False,
        )

        assert [label["name"] for label in ds["labels"]] == ["blood"]
        assert ds["label_vectors"] == [
            [None],
            [1],
        ]
        assert ds["label_masks"] == [
            [0],
            [1],
        ]

    def test_build_image_multilabel_dataset_rejects_empty_annotations(
        self,
    ) -> None:
        # checks empty annotations are not allowed
        with pytest.raises(ValueError, match="has no annotations"):
            build_image_multilabel_dataset(
                dataset_uuid="test_ds",
                annotations=[],
                labelset=self._labelset(),
                treat_unlabeled_as_negative=True,
            )

    def test_build_image_multilabel_dataset_rejects_missing_image_file(
        self,
        tmp_path: Path,
    ) -> None:
        # checks missing image file raises file not found error
        annotations = self._annotations(tmp_path)
        Path(
            annotations[0]["frame"]["file_path"],
            annotations[0]["frame"]["relative_path"],
        ).unlink()

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            build_image_multilabel_dataset(
                dataset_uuid="test_ds",
                annotations=annotations,
                labelset=self._labelset(),
                treat_unlabeled_as_negative=True,
            )

    def test_build_image_multilabel_dataset_rejects_missing_dataset_id(
        self,
        tmp_path: Path,
    ) -> None:
        # checks dataset_id is required for each frame
        annotations = self._annotations(tmp_path)
        annotations[0]["dataset_id"] = None

        with pytest.raises(ValueError, match="Missing or invalid dataset_id"):
            build_image_multilabel_dataset(
                dataset_uuid="test_ds",
                annotations=annotations,
                labelset=self._labelset(),
                treat_unlabeled_as_negative=True,
            )

    def test_build_image_multilabel_dataset_rejects_missing_video_id(
        self,
        tmp_path: Path,
    ) -> None:
        # checks video_id is required for each frame
        annotations = self._annotations(tmp_path)
        annotations[0]["frame"]["video_id"] = None

        with pytest.raises(ValueError, match="Missing or invalid video_id"):
            build_image_multilabel_dataset(
                dataset_uuid="test_ds",
                annotations=annotations,
                labelset=self._labelset(),
                treat_unlabeled_as_negative=True,
            )

    def test_build_image_multilabel_dataset_rejects_missing_frame_id(
        self,
        tmp_path: Path,
    ) -> None:
        # checks frame id is required and must be integer
        annotations = self._annotations(tmp_path)
        annotations[0]["frame"]["id"] = None

        with pytest.raises(ValueError, match="frame.id as int"):
            build_image_multilabel_dataset(
                dataset_uuid="test_ds",
                annotations=annotations,
                labelset=self._labelset(),
                treat_unlabeled_as_negative=True,
            )
