#!/usr/bin/env python3
"""Audit all numeric targets against raw records and sample full-frame image parity."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation
import io

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from convert_lerobot_v3_to_cartesian import UrdfForwardKinematics
from convert_vla_to_lerobot_v3 import read_follower_matrix
from transforms import FRONT, SIDE, validate_info


def verify(root):
    info = json.loads((root / "meta/info.json").read_text())
    processing = json.loads((root / "meta/raw_processing.json").read_text())
    validate_info(info)
    assert processing["smoothing"] is None and processing["roi"] is None
    assert processing["fps"] == 30 and processing["downsample"] is False
    assert all(report["decimated_frames"] == 0 for report in processing["per_episode"])
    assert not processing["gripper_zeroed"] and not processing["static_action_filter"]
    assert processing["gripper_removed"]
    fk = UrdfForwardKinematics(Path(processing["urdf"]))
    sample_episodes = {0, info["total_episodes"]//2, info["total_episodes"]-1}
    total, image_checks, reused_cam3_frames = 0, 0, 0
    for ep, report in enumerate(processing["per_episode"]):
        trial = Path(report["source"])
        path = root / info["data_path"].format(episode_chunk=ep//1000, episode_index=ep)
        table = pq.read_table(path, columns=["observation.state", "action", "timestamp", "episode_index", "frame_index", "index"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        assert len(states) == report["frames"]
        assert np.isfinite(states).all() and np.isfinite(actions).all()
        np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(total, total+len(states)))
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(len(states)))
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), ep)
        np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(len(states))/30, atol=2e-6)
        matrix = read_follower_matrix(trial / "DATA_follower.m")
        with np.load(root / "meta/alignment" / f"episode_{ep:06d}.npz") as source:
            reused_cam3_frames += int(np.count_nonzero(np.diff(source["cam3_frame"]) == 0))
            current, future = matrix[source["robot_row"]], matrix[source["future_robot_row"]]
            positions, rotations = fk.poses(current[:, 2:9])
            next_positions, next_rotations = fk.poses(future[:, 2:9])
            expected_states = np.concatenate((positions, rotations[:, :, 0], rotations[:, :, 1]), axis=1)
            expected_actions = np.concatenate((next_positions-positions,
                Rotation.from_matrix(next_rotations @ rotations.transpose(0, 2, 1)).as_rotvec()), axis=1)
            np.testing.assert_allclose(states, expected_states, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(actions, expected_actions, atol=1e-8, rtol=1e-6)
            np.testing.assert_array_equal(source["future_cam1_frame"]-source["cam1_frame"], 1)
            assert (source["robot_skew_ns"] <= 5_000_000).all()
            assert (source["camera_skew_ns"] <= 25_000_000).all()
            if ep in sample_episodes:
                images = pq.read_table(path, columns=[SIDE, FRONT])
                for camera, key in (("cam1", SIDE), ("cam3", FRONT)):
                    cap = cv2.VideoCapture(str(trial / f"{camera}.mp4"))
                    try:
                        for frame in (0, len(states)//2, len(states)-1):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(source[f"{camera}_frame"][frame]))
                            ok, raw = cap.read()
                            assert ok
                            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                            h, w = rgb.shape[:2]
                            nh, nw = round(h*224/max(h,w)), round(w*224/max(h,w))
                            expected = np.zeros((224,224,3), np.uint8)
                            expected[(224-nh)//2:(224-nh)//2+nh, (224-nw)//2:(224-nw)//2+nw] = cv2.resize(
                                rgb, (nw,nh), interpolation=cv2.INTER_AREA)
                            saved = np.asarray(Image.open(io.BytesIO(images[key][frame].as_py()["bytes"])))
                            np.testing.assert_array_equal(saved, expected)
                            image_checks += 1
                    finally:
                        cap.release()
        total += len(states)
        print(f"Verified {ep+1}/{info['total_episodes']} episodes", flush=True)
    assert total == info["total_frames"] == processing["frames"]
    summary = {"episodes": info["total_episodes"], "frames": total, "fps": info["fps"],
               "consecutive_cam3_nearest_match_reuses": reused_cam3_frames,
               "all_state_action_values_match_raw_fk": True, "full_frame_image_checks": image_checks,
               "gripper_removed": True, "physical_action_dim": 6, "physical_state_dim": 9}
    (root / "meta/verification.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=HERE / "data/threading_tcp6_nosmooth_30hz")
    verify(parser.parse_args().dataset_root.resolve())
