"""Small dependency-free Panda forward kinematics helpers.

The implementation intentionally matches ``convert_lerobot_v3_to_cartesian.py``
so spatial labels and Cartesian action conversion use the same TCP frame.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"expected a 3-vector, got {value!r}")
    return result


def _origin(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


class PandaForwardKinematics:
    """Evaluate the unique Panda base-to-TCP serial chain from a URDF."""

    def __init__(
        self,
        urdf_path: str | Path,
        base_link: str = "panda_link0",
        tcp_link: str = "panda_hand_tcp",
    ) -> None:
        root = ET.parse(Path(urdf_path).expanduser()).getroot()
        by_child: dict[str, dict[str, object]] = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            origin = joint.find("origin")
            axis = joint.find("axis")
            child_name = str(child.attrib["link"])
            by_child[child_name] = {
                "name": str(joint.attrib["name"]),
                "type": str(joint.attrib["type"]),
                "parent": str(parent.attrib["link"]),
                "origin": _origin(
                    _vector(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0)),
                    _vector(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0)),
                ),
                "axis": _vector(
                    axis.attrib.get("xyz") if axis is not None else None,
                    (1, 0, 0),
                ),
            }

        chain: list[dict[str, object]] = []
        link = tcp_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"URDF has no chain from {base_link!r} to {tcp_link!r}")
            item = by_child[link]
            chain.append(item)
            link = str(item["parent"])
        self.chain = list(reversed(chain))
        movable = [item for item in self.chain if item["type"] in {"revolute", "continuous"}]
        if len(movable) != 7:
            raise ValueError(f"expected seven Panda joints, got {[item['name'] for item in movable]}")

    def pose(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(f"expected a finite 7D joint vector, got {q.shape}")
        transform = np.eye(4, dtype=np.float64)
        q_index = 0
        for joint in self.chain:
            transform = transform @ np.asarray(joint["origin"])
            joint_type = str(joint["type"])
            if joint_type in {"revolute", "continuous"}:
                axis = np.asarray(joint["axis"], dtype=np.float64)
                axis /= np.linalg.norm(axis)
                rotation = np.eye(4, dtype=np.float64)
                rotation[:3, :3] = Rotation.from_rotvec(axis * q[q_index]).as_matrix()
                transform = transform @ rotation
                q_index += 1
            elif joint_type != "fixed":
                raise ValueError(f"unsupported joint type {joint_type!r}")
        return transform

    def positions(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != 7:
            raise ValueError(f"expected Nx7 joints, got {joints.shape}")
        return np.stack([self.pose(q)[:3, 3] for q in joints])
