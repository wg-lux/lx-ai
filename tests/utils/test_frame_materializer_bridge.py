# tests/utils/test_frame_materializer_bridge.py

from lx_ai.utils.frame_materializer import ensure_training_frames_available


def test_frame_materializer_reuses_same_frame_path_for_multilabel_annotations(
    monkeypatch, tmp_path
):
    class FakeMaterializer:
        def materialize_frames(
            self,
            *,
            annotation_ids,
            output_root,
            fps,
            ext="jpg",
            quality=2,
            overwrite=False,
        ):
            assert annotation_ids == [101, 102]
            return {
                101: str(tmp_path / "video_2" / "frame_55.jpg"),
                102: str(tmp_path / "video_2" / "frame_55.jpg"),
            }

        def plaintext_processed_video_path(self, video):
            raise AssertionError("not used by frame materialization")

    annotations = [
        {"annotation_id": 101, "label_id": 1, "frame": {}},
        {"annotation_id": 102, "label_id": 2, "frame": {}},
        {},
    ]

    out = ensure_training_frames_available(
        annotations,
        output_root=tmp_path,
        fps=50.0,
        ext="jpg",
        overwrite=False,
        materializer=FakeMaterializer(),
    )

    assert (
        out[0]["frame"]["resolved_frame_path"] == out[1]["frame"]["resolved_frame_path"]
    )
    assert out[0]["label_id"] == 1
    assert out[1]["label_id"] == 2
