from __future__ import annotations

import csv
import json
from pathlib import Path

from lx_ai.data_validation.distribution_report import (
    print_data_validation_report_to_console,
    write_data_validation_report,
)


class TestDistributionReport:
    def _labels(self) -> list[dict]:
        # creates simple label metadata
        return [
            {"id": 1, "name": "polyp"},
            {"id": 2, "name": "blood"},
        ]

    def _dataset(self) -> dict:
        # creates simple multilabel data with known and unknown labels
        return {
            "label_vectors": [
                [1, 0],
                [0, 1],
                [1, None],
                [None, 1],
            ],
            "label_masks": [
                [1, 1],
                [1, 1],
                [1, 0],
                [0, 1],
            ],
            "frame_ids": [10, 11, 12, 13],
            "old_examination_ids": [100, 100, 200, 200],
            "train_indices": [0, 1],
            "val_indices": [2],
            "test_indices": [3],
        }

    def test_write_data_validation_report_creates_files(self, tmp_path: Path) -> None:
        # checks report json and csv files are created
        data = self._dataset()

        json_path, label_csv, exam_csv = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        assert json_path.exists()
        assert label_csv.exists()
        assert exam_csv.exists()

    def test_write_data_validation_report_json_has_expected_top_level_fields(
        self,
        tmp_path: Path,
    ) -> None:
        # checks json report contains expected sections
        data = self._dataset()

        json_path, _, _ = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        report = json.loads(json_path.read_text(encoding="utf-8"))

        assert report["version"] == "1.0"
        assert report["dataset_uuid"] == "test_ds"
        assert report["num_labels"] == 2
        assert "splits" in report
        assert "label_distribution_similarity" in report
        assert "examinations" in report

    def test_write_data_validation_report_has_train_val_test_splits(
        self,
        tmp_path: Path,
    ) -> None:
        # checks report contains train validation and test summaries
        data = self._dataset()

        json_path, _, _ = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        report = json.loads(json_path.read_text(encoding="utf-8"))
        split_names = [split["split_name"] for split in report["splits"]]

        assert split_names == ["train", "val", "test"]

    def test_write_data_validation_report_counts_label_distribution(
        self,
        tmp_path: Path,
    ) -> None:
        # checks positives negatives known and unknown counts are computed
        data = self._dataset()

        json_path, _, _ = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        report = json.loads(json_path.read_text(encoding="utf-8"))
        train = report["splits"][0]

        polyp = train["labels"][0]
        blood = train["labels"][1]

        assert polyp["positives"] == 1
        assert polyp["negatives"] == 1
        assert polyp["known"] == 2
        assert polyp["unknown"] == 0

        assert blood["positives"] == 1
        assert blood["negatives"] == 1
        assert blood["known"] == 2
        assert blood["unknown"] == 0

    def test_write_data_validation_report_handles_empty_split(
        self,
        tmp_path: Path,
    ) -> None:
        # checks empty split does not crash report generation
        data = self._dataset()
        data["test_indices"] = []

        json_path, _, _ = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        report = json.loads(json_path.read_text(encoding="utf-8"))
        test = report["splits"][2]

        assert test["split_name"] == "test"
        assert test["num_samples"] == 0
        assert test["labels"] == []

    def test_label_csv_contains_expected_columns(self, tmp_path: Path) -> None:
        # checks label csv header has all required columns
        data = self._dataset()

        _, label_csv, _ = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        with label_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == [
            "split",
            "label_index",
            "label_id",
            "label_name",
            "positives",
            "negatives",
            "known",
            "unknown",
            "pos_rate",
            "imbalance_ratio",
        ]

    def test_exam_csv_contains_expected_columns(self, tmp_path: Path) -> None:
        # checks exam csv header has all required columns
        data = self._dataset()

        _, _, exam_csv = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        with exam_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == [
            "split",
            "unique_exams",
            "frames",
            "mean_frames_per_exam",
            "max_frames_per_exam",
        ]

    def test_print_data_validation_report_to_console(self, tmp_path: Path, capsys) -> None:
        # checks console printer outputs report sections
        data = self._dataset()

        json_path, _, _ = write_data_validation_report(
            out_dir=tmp_path,
            dataset_uuid="test_ds",
            labels_any=self._labels(),
            **data,
        )

        report = json.loads(json_path.read_text(encoding="utf-8"))
        print_data_validation_report_to_console(report)

        captured = capsys.readouterr()

        assert "DATA VALIDATION REPORT" in captured.out
        assert "LABEL DISTRIBUTION" in captured.out
        assert "SPLIT DISTRIBUTION SIMILARITY" in captured.out
        assert "EXAMINATION DISTRIBUTION" in captured.out