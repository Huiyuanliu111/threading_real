"""Spline-based trajectory data generator for PushBoxEnv (LeRobot v3 output).

Pipeline
--------
1. UI: click up to 5 key waypoints on a top-down schematic of the table.
2. Fit a centripetal Catmull-Rom spline (alpha=0.5) through the waypoints.
3. Collision check the dense path against the fixed obstacles.
   - red = collides (cannot record), green = clear.
4. Execute the path in the simulator (proportional EEF controller) and append
   the episode to a LeRobot v3 dataset (parquet + mp4), matching teleop.py.

This tool is completely independent of teleop.py.

Controls
--------
LMB (on canvas)   add a waypoint (max 5) / drag an existing one
RMB (on canvas)   delete nearest waypoint
N                 clear all waypoints
Enter             if path is collision-free, execute in sim and save episode
Esc               quit (finalizes the LeRobot dataset)

Headless validation (no display / no sim):
    python scripts/datagen.py --validate
"""

import os

# Must run before mujoco/robosuite import (same handling as teleop.py).
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("SDL_VIDEO_X11_FORCE_EGL", "1")

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Task geometry (mirrors envs/pushbox_env.py constants).
from envs.pushbox_env import (
    V2_FIXED_OBSTACLES,
    random_obstacle_configs,
    BOX_INIT_X_RANGE,
    BOX_INIT_Y_RANGE,
)

try:
    from pushbox.lerobot_io import LeRobotWriter

    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

GOAL_POS = (0.0, 0.25)
GOAL_RADIUS = 0.05
WORLD_HALF = 0.4          # table half-size (m) used for canvas bounds
EEF_SAFETY = 0.03         # safety radius (m) inflating obstacles for collision check
RECORD_IMG_SIZE = 96      # image size stored in dataset for training
CONTROL_FREQ = 20         # env control frequency (fps)
MAX_WAYPOINTS = 7         # hard cap on user-selected key points

# LeRobot schema: feature names for info.json (match teleop.py).
FEATURE_NAMES_STATE = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_q1", "gripper_q2"]
FEATURE_NAMES_ACTION = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

# ── canvas geometry ───────────────────────────────────────────────────────────
CANVAS = 600              # clickable schematic size (px)
PANEL = 300               # reference camera panel size (px)
SCALE = CANVAS / (2 * WORLD_HALF)
CX = CY = CANVAS / 2

BG = (28, 28, 32)
WHITE = (240, 240, 240)
GREEN = (80, 200, 80)
RED = (220, 70, 70)
YELLOW = (235, 200, 70)
BLUE = (90, 160, 235)
GRAY = (120, 120, 120)
ORANGE = (220, 140, 60)


# ── 1. Catmull-Rom spline (pure python, per datagen.md) ────────────────────────

def _cr_segment(p0, p1, p2, p3, n, alpha=0.5):
    """Centripetal Catmull-Rom for the P1->P2 segment, n samples in [P1, P2]."""
    def _tj(ti, pi, pj):
        d = float(np.linalg.norm(np.asarray(pj) - np.asarray(pi)))
        return ti + max(d, 1e-6) ** alpha

    t0 = 0.0
    t1 = _tj(t0, p0, p1)
    t2 = _tj(t1, p1, p2)
    t3 = _tj(t2, p2, p3)

    ts = np.linspace(t1, t2, n)
    out = []
    for t in ts:
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        out.append(c)
    return np.asarray(out, dtype=float)


def centripetal_catmull_rom(points, samples_per_seg=24, alpha=0.5):
    """Fit a centripetal Catmull-Rom spline through `points` (N, D) -> (M, D).

    Boundary segments use extrapolated virtual control points:
      P_ext_start = P1 + (P1 - P2),  P_ext_end = PN + (PN - PN-1)
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or len(P) < 2:
        return P.copy()

    start = P[0] + (P[0] - P[1])
    end = P[-1] + (P[-1] - P[-2])
    Pe = np.vstack([start, P, end])  # (N+2, D)

    segs = []
    for i in range(1, len(Pe) - 2):
        seg = _cr_segment(Pe[i - 1], Pe[i], Pe[i + 1], Pe[i + 2], samples_per_seg, alpha)
        if i > 1:
            seg = seg[1:]  # drop duplicated knot shared with previous segment
        segs.append(seg)
    return np.vstack(segs)


# ── 2. Collision detection ─────────────────────────────────────────────────────

def path_collides(path_xy, obstacles=None, safety=EEF_SAFETY):
    """True if any path point falls inside an (inflated) obstacle AABB."""
    obstacles = obstacles if obstacles is not None else V2_FIXED_OBSTACLES
    path_xy = np.asarray(path_xy, dtype=float)
    for ob in obstacles:
        ox, oy = ob["pos"]
        hx = ob["half_size"][0] + safety
        hy = ob["half_size"][1] + safety
        inside = (np.abs(path_xy[:, 0] - ox) <= hx) & (np.abs(path_xy[:, 1] - oy) <= hy)
        if inside.any():
            return True
    return False


def path_out_of_bounds(path_xy, half=WORLD_HALF):
    path_xy = np.asarray(path_xy, dtype=float)
    return bool((np.abs(path_xy) > half).any())


# ── coordinate mapping (world <-> canvas px) ───────────────────────────────────
# Schematic top-down view: world +x -> up, world +y -> right.

def world_to_canvas(wx, wy):
    sx = CX + wy * SCALE
    sy = CY - wx * SCALE
    return int(round(sx)), int(round(sy))


def canvas_to_world(sx, sy):
    wy = (sx - CX) / SCALE
    wx = -(sy - CY) / SCALE
    return float(wx), float(wy)


# ── 3. UI drawing helpers ──────────────────────────────────────────────────────

def _draw_overlay(screen, pygame, waypoints, dense, collides, box_pos=None, obstacles=None):
    """Draw obstacles, goal, box-init region, waypoints and spline on the canvas."""
    obstacles = obstacles if obstacles is not None else V2_FIXED_OBSTACLES
    for ob in obstacles:
        ox, oy = ob["pos"]
        hx, hy = ob["half_size"][0], ob["half_size"][1]
        corners = [
            world_to_canvas(ox - hx, oy - hy),
            world_to_canvas(ox - hx, oy + hy),
            world_to_canvas(ox + hx, oy + hy),
            world_to_canvas(ox + hx, oy - hy),
        ]
        pygame.draw.polygon(screen, ORANGE, corners)
        pygame.draw.polygon(screen, YELLOW, corners, 2)

    bx0, bx1 = BOX_INIT_X_RANGE
    by0, by1 = BOX_INIT_Y_RANGE
    region = [
        world_to_canvas(bx0, by0),
        world_to_canvas(bx0, by1),
        world_to_canvas(bx1, by1),
        world_to_canvas(bx1, by0),
    ]
    pygame.draw.polygon(screen, BLUE, region, 2)

    gc = world_to_canvas(*GOAL_POS)
    pygame.draw.circle(screen, GREEN, gc, int(GOAL_RADIUS * SCALE), 2)

    # current box position
    if box_pos is not None:
        bx, by = float(box_pos[0]), float(box_pos[1])
        half_px = max(2, int(0.025 * SCALE))  # box half-size 2.5cm -> px
        box_corners = [
            world_to_canvas(bx - 0.025, by - 0.025),
            world_to_canvas(bx - 0.025, by + 0.025),
            world_to_canvas(bx + 0.025, by + 0.025),
            world_to_canvas(bx + 0.025, by - 0.025),
        ]
        pygame.draw.polygon(screen, (255, 100, 100), box_corners)
        pygame.draw.polygon(screen, RED, box_corners, 2)

    if dense is not None and len(dense) >= 2:
        col = RED if collides else GREEN
        pts = [world_to_canvas(p[0], p[1]) for p in dense]
        pygame.draw.lines(screen, col, False, pts, 3)

    for wx, wy in waypoints:
        c = world_to_canvas(wx, wy)
        pygame.draw.circle(screen, WHITE, c, 6)
        pygame.draw.circle(screen, GRAY, c, 6, 1)


def _capture_cam_float(env, camera_name, size):
    """Render a camera RGB frame -> (size, size, 3) float32 in [0,1] (UI panels)."""
    if hasattr(env.sim, "_render_context_offscreen") and env.sim._render_context_offscreen is not None:
        try:
            env.sim._render_context_offscreen.gl_ctx.make_current()
        except Exception:
            pass
    frame = env.sim.render(width=size, height=size, camera_name=camera_name)
    return np.flipud(frame).astype(np.float32) / 255.0


def _capture_cam_uint8(env, camera_name, size=RECORD_IMG_SIZE):
    """Render a camera RGB frame -> (size, size, 3) uint8 (LeRobot video)."""
    if hasattr(env.sim, "_render_context_offscreen") and env.sim._render_context_offscreen is not None:
        try:
            env.sim._render_context_offscreen.gl_ctx.make_current()
        except Exception:
            pass
    frame = env.sim.render(width=size, height=size, camera_name=camera_name)
    return np.flipud(frame).astype(np.uint8)


def _frame_to_surface(pygame, frame01, target):
    """float32 [0,1] (H,W,3) -> scaled pygame Surface (target,target)."""
    arr = (np.clip(frame01, 0.0, 1.0) * 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
    if surf.get_size() != (target, target):
        surf = pygame.transform.smoothscale(surf, (target, target))
    return surf


# ── 4. Simulator executor + LeRobot recording ──────────────────────────────────

def _get_lerobot_features() -> dict:
    """Build LeRobot v3 features dict matching PushBox data schema (== teleop)."""
    return {
        "observation.images.top45": {
            "dtype": "video",
            "shape": (RECORD_IMG_SIZE, RECORD_IMG_SIZE, 3),
        },
        "observation.images.sideview": {
            "dtype": "video",
            "shape": (RECORD_IMG_SIZE, RECORD_IMG_SIZE, 3),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": FEATURE_NAMES_STATE,
        },
        "observation.box_pos": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["box_x", "box_y"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": FEATURE_NAMES_ACTION,
        },
    }


def _make_lerobot_frame(env, obs, action):
    """Build one LeRobot frame dict from current obs + action."""
    joints = obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32))
    gripper = obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
    agent_state = np.concatenate([joints, gripper]).astype(np.float32)
    box_pos = obs.get("box_pos_xy", np.zeros(2, dtype=np.float32)).astype(np.float32)
    return {
        "observation.images.top45": _capture_cam_uint8(env, "top45"),
        "observation.images.sideview": _capture_cam_uint8(env, "sideview"),
        "observation.state": agent_state,
        "observation.box_pos": box_pos,
        "action": action.astype(np.float32),
        "task": "push_box",
    }


def execute_path(env, dense_xy, frames, kp=32.0, tol=0.01, max_substeps=8,
                 max_total=5000):
    """Drive the EEF through dense_xy (world frame), appending LeRobot frames.

    Dense spline points are used as a path guide, but the controller skips
    ahead to the next meaningful target each step — so spline density does
    NOT determine execution speed.  Speed is governed purely by kp·distance.

    Returns (success, info). z and gripper(closed) are held constant.
    """
    obs = env._last_obs
    total = 0
    idx = 0  # next spline point to aim for

    while idx < len(dense_xy) and total < max_total:
        eef = obs["robot0_eef_pos"].astype(np.float32)
        target = np.asarray(dense_xy[idx], dtype=np.float32)
        diff_xy = target - eef[:2]
        dist = np.linalg.norm(diff_xy)

        # Skip points we are already almost on top of
        while dist < 0.003 and idx < len(dense_xy) - 1:
            idx += 1
            target = np.asarray(dense_xy[idx], dtype=np.float32)
            diff_xy = target - eef[:2]
            dist = np.linalg.norm(diff_xy)

        # Arrived at or past last point
        if idx >= len(dense_xy) - 1 and dist < 0.015:
            break

        d = np.clip(kp * diff_xy, -1.0, 1.0)
        action = np.array([d[0], d[1], 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        frames.append(_make_lerobot_frame(env, obs, action))
        env._last_obs = obs
        total += 1

        if done:
            return bool(info.get("success", False)), info

    success = env._check_success() if hasattr(env, "_check_success") else False
    return bool(success), {"success": success}


# ── 5. Main: env + EGL + pygame loop ───────────────────────────────────────────

def _make_env(horizon):
    _display = os.environ.pop("DISPLAY", None)
    from envs import PushBoxEnv
    env = PushBoxEnv(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=horizon,
        hard_reset=False,
        camera_names="top45,sideview,birdview",
        camera_heights=256,
        camera_widths=256,
    )
    if _display is not None:
        os.environ["DISPLAY"] = _display
    return env


def _release_egl():
    try:
        from robosuite.renderers.context import egl_context as ec
        ec.EGL.eglMakeCurrent(ec.EGL_DISPLAY, ec.EGL.EGL_NO_SURFACE,
                              ec.EGL.EGL_NO_SURFACE, ec.EGL.EGL_NO_CONTEXT)
    except Exception:
        pass


def run_ui(args):
    if not HAS_LEROBOT:
        print("[ERROR] pushbox.lerobot_io not importable; cannot write dataset", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    features = _get_lerobot_features()
    writer = LeRobotWriter.resume_or_create(
        repo_id=args.repo_id,
        fps=CONTROL_FREQ,
        features=features,
        use_videos=True,
        root=str(out_dir),
    )

    env = _make_env(args.horizon)
    env.deterministic_reset = False  # ensure random box placement
    rng = np.random.default_rng()
    obstacles = random_obstacle_configs(rng)
    env.obstacle_configs = obstacles
    obs, grip_steps = env.init_with_grip()
    env._last_obs = obs
    print(f"[datagen] env ready, auto-grip {grip_steps} steps")

    _release_egl()
    import pygame
    pygame.init()
    win_w, win_h = CANVAS + PANEL + 20, CANVAS + 50
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("PushBox DataGen — spline waypoints (LeRobot)")
    font = pygame.font.SysFont("monospace", 15, bold=True)
    sm = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()

    waypoints = []          # list[(wx, wy)], max MAX_WAYPOINTS
    drag_idx = None
    saved = 0
    status_msg = f"click <= {MAX_WAYPOINTS} waypoints; Enter=record  N=clear  Esc=quit"

    def compute_dense():
        if len(waypoints) < 2:
            return None
        return centripetal_catmull_rom(np.array(waypoints), args.samples_per_seg)

    def nearest_wp(sx, sy, thresh=12):
        best, bd = None, thresh
        for i, (wx, wy) in enumerate(waypoints):
            cx, cy = world_to_canvas(wx, wy)
            d = np.hypot(cx - sx, cy - sy)
            if d < bd:
                best, bd = i, d
        return best

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_n:
                    waypoints.clear()
                    status_msg = "cleared"
                elif ev.key == pygame.K_RETURN:
                    dense = compute_dense()
                    if dense is None:
                        status_msg = "need >= 2 waypoints"
                    elif path_collides(dense, obstacles):
                        status_msg = "COLLISION — path rejected"
                    elif path_out_of_bounds(dense):
                        status_msg = "OUT OF BOUNDS — path rejected"
                    else:
                        status_msg = "RECORDING..."
                        # Show recording overlay before blocking
                        screen.fill(BG)
                        screen.blit(font.render("RECORDING... please wait", True, YELLOW), (10, CANVAS + 10))
                        screen.blit(sm.render(f"waypoints: {len(waypoints)}/{MAX_WAYPOINTS}   saved: {saved}", True, WHITE), (10, CANVAS + 32))
                        pygame.display.flip()
                        frames = []
                        try:
                            # Re-bind EGL context before offscreen rendering in execute_path.
                            if hasattr(env.sim, "_render_context_offscreen") and env.sim._render_context_offscreen is not None:
                                env.sim._render_context_offscreen.gl_ctx.make_current()
                            env.deterministic_reset = False  # randomize box each episode
                            env.obstacle_configs = obstacles
                            obs, _gs = env.init_with_grip()
                            env._last_obs = obs
                            # prepend current EEF xy so the robot moves from
                            # the box's random init position to the first waypoint.
                            # Interpolate intermediate points so the initial segment
                            # has the same density as the rest of the spline.
                            start_xy = obs["robot0_eef_pos"][:2].astype(np.float32)
                            first_wp = dense[0][:2]
                            dist_to_first = np.linalg.norm(first_wp - start_xy)
                            n_interp = max(2, int(dist_to_first * args.samples_per_seg / 0.3))
                            ramp = np.linspace(start_xy, first_wp, n_interp + 1)
                            dense_with_start = np.vstack([ramp, dense[1:]])
                            success, info = execute_path(env, dense_with_start, frames,
                                                         max_total=args.horizon)
                            if frames:
                                for fr in frames:
                                    writer.add_frame(fr)
                                ep_idx = writer.save_episode()
                                writer._flush_meta()
                                saved += 1
                                status_msg = f"saved ep {ep_idx} ({len(frames)} steps) success={success}"
                                waypoints.clear()
                                # randomise obstacles for next episode
                                obstacles = random_obstacle_configs(rng)
                                env.obstacle_configs = obstacles
                                obs, _gs = env.init_with_grip()
                                env._last_obs = obs
                            else:
                                status_msg = "no steps recorded"
                        except Exception as e:
                            status_msg = f"recording failed: {e}"
                            print(f"[datagen] recording error: {e}", file=sys.stderr)
                            import traceback
                            traceback.print_exc()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                sx, sy = ev.pos
                if sx < CANVAS and sy < CANVAS:
                    if ev.button == 1:
                        idx = nearest_wp(sx, sy)
                        if idx is not None:
                            drag_idx = idx
                        elif len(waypoints) < MAX_WAYPOINTS:
                            waypoints.append(canvas_to_world(sx, sy))
                        else:
                            status_msg = f"max {MAX_WAYPOINTS} waypoints reached"
                    elif ev.button == 3:
                        idx = nearest_wp(sx, sy)
                        if idx is not None:
                            waypoints.pop(idx)
            elif ev.type == pygame.MOUSEBUTTONUP:
                drag_idx = None
            elif ev.type == pygame.MOUSEMOTION:
                if drag_idx is not None:
                    sx, sy = ev.pos
                    sx = min(sx, CANVAS); sy = min(sy, CANVAS)
                    waypoints[drag_idx] = canvas_to_world(sx, sy)

        dense = compute_dense()
        collides = dense is not None and (path_collides(dense, obstacles) or path_out_of_bounds(dense))

        # ── draw ──
        screen.fill(BG)
        canvas_rect = pygame.Rect(0, 0, CANVAS, CANVAS)
        pygame.draw.rect(screen, (16, 16, 18), canvas_rect)
        box_pos = env._last_obs.get("box_pos_xy", None) if env._last_obs is not None else None
        _draw_overlay(screen, pygame, waypoints, dense, collides, box_pos, obstacles)
        pygame.draw.rect(screen, GRAY, canvas_rect, 1)

        try:
            top45 = _capture_cam_float(env, "top45", 128)
            side = _capture_cam_float(env, "sideview", 128)
            screen.blit(_frame_to_surface(pygame, top45, PANEL), (CANVAS + 14, 0))
            screen.blit(_frame_to_surface(pygame, side, PANEL), (CANVAS + 14, PANEL + 10))
        except Exception as e:
            screen.blit(sm.render(f"cam err: {e}", True, RED), (CANVAS + 14, 10))

        bar_y = CANVAS + 6
        col = RED if collides else (GREEN if dense is not None else WHITE)
        screen.blit(font.render(status_msg, True, col), (8, bar_y))
        screen.blit(sm.render(f"waypoints: {len(waypoints)}/{MAX_WAYPOINTS}   saved: {saved}",
                              True, WHITE), (8, bar_y + 22))

        pygame.display.flip()
        clock.tick(30)

    writer.finalize()
    env.close()
    pygame.quit()
    print(f"[datagen] {saved} episodes written to LeRobot dataset at {out_dir}/")


# ── validation (no display / no sim) ───────────────────────────────────────────

def run_validate():
    print("[validate] spline + collision sanity checks")
    wps = np.array([[0.0, -0.20], [0.0, 0.0], [0.0, 0.20]])
    dense = centripetal_catmull_rom(wps, samples_per_seg=20)
    assert dense.ndim == 2 and dense.shape[1] == 2, dense.shape
    assert np.allclose(dense[0], wps[0], atol=1e-6), dense[0]
    assert np.allclose(dense[-1], wps[-1], atol=1e-6), dense[-1]
    print(f"  OK spline shape {dense.shape}, endpoints preserved")

    for w in wps[1:-1]:
        d = np.linalg.norm(dense - w, axis=1).min()
        assert d < 1e-3, f"waypoint {w} not on spline (min dist {d})"
    print("  OK spline passes through all waypoints")

    ob = V2_FIXED_OBSTACLES[0]
    through = np.array([[ob["pos"][0], ob["pos"][1] - 0.3],
                        [ob["pos"][0], ob["pos"][1]],
                        [ob["pos"][0], ob["pos"][1] + 0.3]])
    dense_c = centripetal_catmull_rom(through, 20)
    assert path_collides(dense_c), "expected collision through obstacle"
    print("  OK collision detected through obstacle")

    clear = np.array([[0.0, -0.20], [-0.10, 0.0], [0.0, 0.20]])
    dense_ok = centripetal_catmull_rom(clear, 20)
    print(f"  corridor path collides={path_collides(dense_ok)} (geometry-dependent)")

    for wx, wy in [(0.1, -0.2), (-0.3, 0.15)]:
        sx, sy = world_to_canvas(wx, wy)
        rx, ry = canvas_to_world(sx, sy)
        assert abs(rx - wx) < 1e-2 and abs(ry - wy) < 1e-2, (wx, wy, rx, ry)
    print("  OK world<->canvas round-trip")

    feats = _get_lerobot_features()
    assert set(feats) == {
        "observation.images.top45", "observation.images.sideview",
        "observation.state", "observation.box_pos", "action",
    }, feats.keys()
    print(f"  OK LeRobot features schema: {sorted(feats)}")
    print(f"  OK MAX_WAYPOINTS={MAX_WAYPOINTS}")
    print("[validate] all checks passed")


def main():
    ap = argparse.ArgumentParser(description="Spline waypoint data generator (LeRobot v3) for PushBox")
    ap.add_argument("--out", default="data/datagen", help="output dir for LeRobot dataset")
    ap.add_argument("--repo-id", default="pushbox/datagen",
                    help="LeRobot dataset repo_id")
    ap.add_argument("--horizon", type=int, default=5000, help="max steps per episode")
    ap.add_argument("--samples-per-seg", type=int, default=24,
                    help="spline samples per control-point segment")
    ap.add_argument("--validate", action="store_true",
                    help="run headless math checks (no UI / no sim)")
    args = ap.parse_args()

    if args.validate:
        run_validate()
        return
    run_ui(args)


if __name__ == "__main__":
    main()
