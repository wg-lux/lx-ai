# tests/utils/test_frame_materializer_bridge.py

from lx_ai.utils.frame_materializer import ensure_training_frames_available


def test_frame_materializer_reuses_same_frame_path_for_multilabel_annotations(
    monkeypatch, tmp_path
):
    def fake_materialize_frames_for_lxai_annotations(
        *, annotation_ids, output_root, fps, ext, overwrite
    ):
        assert annotation_ids == [101, 102]
        return {
            101: str(tmp_path / "video_2" / "frame_55.jpg"),
            102: str(tmp_path / "video_2" / "frame_55.jpg"),
        }

    import lx_ai.utils.endoregdb_encrypted_frame_bridge as bridge

    monkeypatch.setattr(
        bridge,
        "materialize_frames_for_lxai_annotations",
        fake_materialize_frames_for_lxai_annotations,
    )

    annotations = [
        {"annotation_id": 101, "label_id": 1, "frame": {}},
        {"annotation_id": 102, "label_id": 2, "frame": {}},
    ]

    out = ensure_training_frames_available(
        annotations,
        output_root=tmp_path,
        fps=50.0,
        ext="jpg",
        overwrite=False,
    )

    assert (
        out[0]["frame"]["resolved_frame_path"] == out[1]["frame"]["resolved_frame_path"]
    )
    assert out[0]["label_id"] == 1
    assert out[1]["label_id"] == 2
