"""Run with python -m pytest threading_real/pi05/visual/test_crop.py."""
import copy
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from threading_real.pi05.visual.crop import load, save, transform, validate, make_reader, make_observation_adapter


@pytest.fixture
def config():
    return {"version": 1, "output_size": 224, "resize": "letterbox", "cameras": {
        "cam1": {"source_size": [640, 480], "roi": [100, 80, 324, 304]},
        "cam3": {"source_size": [640, 480], "roi": [20, 40, 468, 264]},
    }}


def test_crop_retains_pixels_and_letterboxes(config):
    image = np.zeros((480, 640, 3), np.uint8)
    image[80:304, 100:324] = [23, 77, 201]
    assert np.array_equal(transform(image, "cam1", config), image[80:304, 100:324])
    image[:] = [23, 77, 201]
    actual = transform(image, "cam3", config)
    assert actual.shape == (224, 224, 3)
    assert (actual[:56] == 0).all() and (actual[168:] == 0).all()
    assert (actual[56:168] == [23, 77, 201]).all()


def test_rejects_resized_input_and_bad_config(config):
    with pytest.raises(ValueError, match="resolution"):
        transform(np.zeros((224, 224, 3), np.uint8), "cam1", config)
    for box in ([0, 0, 0, 4], [-1, 0, 20, 20], [0, 0, 641, 480], [0.5, 0, 20, 20]):
        bad = copy.deepcopy(config)
        bad["cameras"]["cam1"]["roi"] = box
        with pytest.raises(ValueError):
            validate(bad)


def test_converter_and_live_path_match_without_recropping(config):
    import convert_vla_to_lerobot_v3 as converter
    rgb = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    class Capture:
        def read(self):
            return True, rgb[..., ::-1].copy()
    reader = make_reader(converter.SequentialVideoReader, config)(Capture(), Path("cam1.mp4"))
    converted_rgb = cv2.cvtColor(reader.read(0), cv2.COLOR_BGR2RGB)
    assert np.array_equal(converted_rgb, cv2.cvtColor(reader.read(0), cv2.COLOR_BGR2RGB))
    observe = make_observation_adapter(lambda *args, **kwargs: args, config)
    live = observe(rgb, None, rgb, [0] * 7, 0.02, 224)
    assert np.array_equal(converted_rgb, live[0])
    assert np.array_equal(live[2], transform(rgb, "cam3", config))
    with pytest.raises(ValueError, match="pre-resize"):
        observe(rgb, None, rgb, [0] * 7, 0.02, 224, 96)


def test_preview_video_and_metadata_roundtrip(config, tmp_path):
    for camera in config["cameras"]:
        writer = cv2.VideoWriter(str(tmp_path / f"{camera}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
        assert writer.isOpened()
        for intensity in (30, 100, 200):
            writer.write(np.full((480, 640, 3), intensity, np.uint8))
        writer.release()
    path = tmp_path / "config.json"
    command = [sys.executable, str(REPO / "threading_real/pi05/visual/preview.py"),
               "--episode", str(tmp_path), "--frames", "0", "2",
               "--cam1-roi", "100", "80", "324", "304",
               "--cam3-roi", "20", "40", "468", "264",
               "--save-config", str(path), "--output", str(tmp_path / "preview")]
    subprocess.run(command, check=True, capture_output=True)
    assert load(path) == config
    assert len(list((tmp_path / "preview").glob("*.png"))) == 16
    assert cv2.imread(str(tmp_path / "preview/cam1_000002_input224.png")).shape == (224, 224, 3)
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_pipeline_passes_config_and_keeps_failed_intermediates(config, tmp_path, monkeypatch):
    from threading_real.pi05.training import build_dataset as pipeline
    config_path = tmp_path / "crop.json"
    save(config, config_path)
    output = tmp_path / "dataset"
    def stage(roots, path):
        path.mkdir()
        return 80
    monkeypatch.setattr(pipeline, "stage_raw_roots", stage)
    calls = []
    def run(command):
        calls.append(command)
        if command[1].endswith("prepare_dataset.py"):
            (output / "meta").mkdir(parents=True)
    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(sys, "argv", ["build", "--raw-root", str(tmp_path),
                                      "--output", str(output), "--visual-config", str(config_path)])
    pipeline.main()
    assert calls[0][1].endswith("convert_raw_cropped.py")
    assert calls[0][-2:] == ["--visual-config", str(config_path)]
    assert load(output / "meta/visual_preprocessing.json") == config
    assert "--drop-zero-actions" not in calls[3]
    # On a later preflight failure, keep expensive conversion outputs for inspection.
    output2 = tmp_path / "failed"
    intermediate = tmp_path / "failed.joint_source"
    def fail(command):
        intermediate.mkdir(exist_ok=True)
        if command[1].endswith("prepare_dataset.py"):
            (output2 / "meta").mkdir(parents=True)
        if command[1].endswith("preflight.py"):
            raise RuntimeError("preflight failed")
    monkeypatch.setattr(pipeline, "run", fail)
    monkeypatch.setattr(sys, "argv", ["build", "--raw-root", str(tmp_path), "--output", str(output2)])
    with pytest.raises(RuntimeError, match="preflight"):
        pipeline.main()
    assert intermediate.is_dir()


def test_deployment_loads_checkpoint_config_and_restores_hooks(config, tmp_path, monkeypatch):
    from threading_real.pi05.deployment import cropped
    save(config, tmp_path / "visual_preprocessing.json")
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05", "input_features": {
        "observation.state": {"shape": [8]}}}))
    (tmp_path / "state_representation.json").write_text(json.dumps({
        "state_representation": "joint", "state_dim": 8}))
    args = argparse.Namespace(checkpoint=tmp_path, policy_kind="auto", image_size=None,
                              pre_resize_image_size=None, chunk_selector=None)
    runner = types.ModuleType("threading_real.scripts.deployment.cartesian")
    runner.build_parser = lambda: types.SimpleNamespace(parse_args=lambda: args)
    import torch
    original_observe = lambda *a, **k: types.SimpleNamespace(
        images=a, agent_pos=torch.tensor([*a[3], a[4]], dtype=torch.float32))
    original_rig = lambda *a, **k: k
    runner.make_observation, runner.RealSenseRig = original_observe, original_rig
    def main():
        assert runner.RealSenseRig(fps=30) == {"fps": 30, "width": 640, "height": 480}
        raw = np.zeros((480, 640, 3), np.uint8)
        obs = runner.make_observation(raw, None, raw, [0] * 7, 0.02, 224)
        assert obs.images[0].shape == (224, 224, 3)
        return 0
    runner.main = main
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    assert cropped.main() == 0
    assert runner.make_observation is original_observe and runner.RealSenseRig is original_rig
    args.chunk_selector = tmp_path
    with pytest.raises(ValueError, match="selector"):
        cropped.main()
    args.chunk_selector = None
    (tmp_path / "visual_preprocessing.json").unlink()
    with pytest.raises(FileNotFoundError):
        cropped.main()


def test_raw_conversion_entry_installs_only_temporary_reader(config, tmp_path, monkeypatch):
    from threading_real.pi05.training import convert_raw_cropped as entry
    path = tmp_path / "crop.json"
    save(config, path)
    output = tmp_path / "dataset"
    original = entry.converter.SequentialVideoReader
    monkeypatch.setattr(sys, "argv", ["convert", str(tmp_path), str(output),
                                     "--visual-config", str(path), "--image-size", "224", "--skip-depth"])
    def convert(args):
        assert entry.converter.SequentialVideoReader is not original
        assert args.skip_depth
        (output / "meta").mkdir(parents=True)
        return {"output": str(output)}
    monkeypatch.setattr(entry.converter, "convert", convert)
    assert entry.main() == 0
    assert entry.converter.SequentialVideoReader is original
    assert load(output / "meta/visual_preprocessing.json") == config
