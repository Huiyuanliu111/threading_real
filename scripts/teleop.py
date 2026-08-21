"""Teleoperation UI for PushBoxEnv — keyboard control + episode recording.

Controls
--------
W / S          EEF +X / −X  (back / forward on table)
A / D          EEF +Y / −Y  (left / right)
Q / E          EEF +Z / −Z  (up / down)
Z / C          EEF yaw about vertical axis (− / +)
Arrow ↑↓←→     same as WASD (alternative)
Space          toggle gripper open/close
F11            toggle fullscreen
Tab / Shift+Tab  next / previous view
1–4            birdview / agentview / frontview / sideview
5 / 6          top45 / sideview (ARP recording views)
0              orbit free cam (+/− zoom, LMB rotate, RMB pan)
N              discard current episode and reset
Enter          save current episode to disk and reset
Esc            quit (unsaved in-progress data is discarded)

Recording starts automatically on each new episode.

Saved data (LeRobot v3 format)
------------------------------
Each recorded episode is saved as part of a LeRobot v3 dataset at <output_dir>:
  observation.images.top45    (video)  96×96 RGB from top45 camera
  observation.images.sideview (video)  96×96 RGB from sideview camera
  observation.state           (9,)     float32  robot state (7 joints + 2 gripper qpos)
  action                      (7,)     float32  OSC_POSE [dx,dy,dz,dr,dp,dy,grip]

Directory structure:
  <out_dir>/
    meta/
      info.json              dataset schema & FPS
      episodes/chunk-000/    per-episode metadata (parquet)
      stats.json             normalization statistics
      tasks.parquet          task definitions
    data/chunk-000/          tabular data (parquet)
    videos/
      observation.images.top45/chunk-000/    mp4 video shards
      observation.images.sideview/chunk-000/ mp4 video shards
"""

import os

# Must run before mujoco/robosuite import. setdefault() fails if shell has MUJOCO_GL=glx.
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
# SDL2 X11 defaults to GLX; conflicts with MuJoCo EGL on NVIDIA. Force SDL onto EGL.
os.environ.setdefault("SDL_VIDEO_X11_FORCE_EGL", "1")

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    from pushbox.lerobot_io import LeRobotWriter

    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

# pygame / PushBoxEnv imported in main() AFTER DISPLAY is hidden for EGL init.

# ── constants ────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 480, 480          # offscreen render resolution
RECORD_IMG_SIZE = 96             # image size stored in dataset for ARP training
CONTROL_FREQ = 20                # env control frequency (fps)

STEP = 0.6        # action magnitude per key (fraction of max output)
YAW_STEP = 0.6    # gripper yaw about vertical axis (OSC index 5)
LIFT_DZ = 0.1     # per-step Z displacement during post-close lift
CLOSE_DURATION = 25   # steps to fully close gripper before lifting

# LeRobot schema: feature names for info.json
FEATURE_NAMES_STATE = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_q1", "gripper_q2"]
FEATURE_NAMES_ACTION = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

BG      = (30,  30,  30)
WHITE   = (240, 240, 240)
GREEN   = ( 80, 200,  80)
RED     = (220,  60,  60)
YELLOW  = (230, 200,  50)
GRAY    = (120, 120, 120)

# Fixed cameras in robosuite TableArena (table_arena.xml)
CAMERA_VIEWS = ("birdview", "agentview", "frontview", "sideview", "top45")
VIEW_CYCLE = ("orbit",) + CAMERA_VIEWS

ORBIT_SENS = 0.25
ZOOM_STEP = 1.12
PAN_SENS = 0.0012


class WindowLayout:
    """Window / camera panel geometry (windowed or fullscreen)."""

    def __init__(self):
        self.fullscreen = False
        self.win_w, self.win_h = 640, 480
        self.cam_w, self.cam_h = 480, 480

    def refresh(self):
        import pygame

        if self.fullscreen:
            info = pygame.display.Info()
            self.win_w, self.win_h = info.current_w, info.current_h
        else:
            self.win_w, self.win_h = 640, 480
        self.cam_w = max(CAM_W, int(self.win_w * 0.75))
        self.cam_h = self.win_h

    def set_mode(self):
        import pygame

        flags = pygame.FULLSCREEN if self.fullscreen else 0
        if not self.fullscreen:
            flags |= pygame.SWSURFACE
        return pygame.display.set_mode((self.win_w, self.win_h), flags)


class OrbitCamera:
    """MuJoCo free-camera orbit around the table centre."""

    def __init__(self):
        self.lookat = np.array([0.0, 0.0, 0.85], dtype=np.float64)
        self.distance = 1.6
        self.azimuth = 120.0
        self.elevation = -25.0

    def rotate(self, dx: float, dy: float) -> None:
        self.azimuth -= dx * ORBIT_SENS
        self.elevation = float(np.clip(self.elevation - dy * ORBIT_SENS, -89.0, 89.0))

    def zoom_wheel(self, steps: int) -> None:
        for _ in range(abs(steps)):
            if steps > 0:
                self.distance = max(0.35, self.distance / ZOOM_STEP)
            else:
                self.distance = min(5.0, self.distance * ZOOM_STEP)

    def pan(self, dx: float, dy: float) -> None:
        """Pan lookat in the view plane (RMB drag)."""
        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        cos_el = np.cos(el)
        view = np.array([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)])
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(view, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn
        up = np.cross(right, view)
        up /= np.linalg.norm(up) + 1e-9
        s = self.distance * PAN_SENS
        self.lookat -= right * dx * s
        self.lookat += up * dy * s

    def apply(self, cam) -> None:
        import mujoco
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        cam.distance = self.distance
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation


# ── helpers ──────────────────────────────────────────────────────────────────

def keys_to_action(pressed, gripper_open: bool) -> np.ndarray:
    """Build a 7-D OSC_POSE action from currently held keys."""
    import pygame

    dx = dy = dz = 0.0

    if pressed[pygame.K_w] or pressed[pygame.K_UP]:     dx = -STEP
    if pressed[pygame.K_s] or pressed[pygame.K_DOWN]:   dx =  STEP
    if pressed[pygame.K_a] or pressed[pygame.K_LEFT]:   dy = -STEP
    if pressed[pygame.K_d] or pressed[pygame.K_RIGHT]:  dy =  STEP
    if pressed[pygame.K_q]:                             dz =  STEP
    if pressed[pygame.K_e]:                             dz = -STEP

    dyaw = 0.0
    if pressed[pygame.K_z]:                             dyaw = -YAW_STEP
    if pressed[pygame.K_c]:                             dyaw =  YAW_STEP

    gripper = -1.0 if gripper_open else 1.0  # robosuite: -1=open, 1=close
    return np.array([dx, dy, dz, 0.0, 0.0, dyaw, gripper], dtype=np.float32)


def reset_env(env):
    """Reset env, auto-grip the box, and restart step counter.

    Grip steps do NOT count toward the episode horizon.
    """
    obs, grip_steps = env.init_with_grip()
    print(f"[init] auto-grip completed in {grip_steps} steps (not counted toward horizon)")
    return obs


def capture_cam(env, camera_name: str, size: int = RECORD_IMG_SIZE) -> np.ndarray:
    """Render camera RGB frame for dataset recording, shape (H, W, 3) uint8 [0,255]."""
    frame = env.sim.render(width=size, height=size, camera_name=camera_name)
    frame = np.flipud(frame).astype(np.uint8)
    return frame


class EpisodeFrameBuffer:
    """Accumulates one episode of frames for LeRobot recording."""

    def __init__(self):
        self.frames: list[dict] = []

    def append(self, frame: dict):
        self.frames.append(frame)

    def __len__(self):
        return len(self.frames)

    def clear(self):
        self.frames.clear()


def _get_lerobot_features() -> dict:
    """Build LeRobot v3 features dict matching PushBox data schema."""
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


def save_episode_lerobot(writer, buf: EpisodeFrameBuffer, ep_idx: int) -> None:
    """Replay buffered frames into LeRobot writer and save the episode."""
    for frame in buf.frames:
        writer.add_frame(frame)
    writer.save_episode()
    print(f"[saved] episode {ep_idx} ({len(buf)} steps) → LeRobot dataset")
    buf.clear()


# ── rendering helpers ─────────────────────────────────────────────────────────

def _pixels_to_surface(frame):
    import pygame
    frame = np.flipud(frame)
    return pygame.surfarray.make_surface(frame.transpose(1, 0, 2))


def render_frame_fixed(env, camera_name: str):
    frame = env.sim.render(width=CAM_W, height=CAM_H, camera_name=camera_name)
    return _pixels_to_surface(frame)


def render_frame_orbit(env, orbit: OrbitCamera):
    rc = env.sim._render_context_offscreen
    orbit.apply(rc.cam)
    rc.render(width=CAM_W, height=CAM_H, camera_id=-1)
    frame = rc.read_pixels(CAM_W, CAM_H, depth=False)
    return _pixels_to_surface(frame)


def _scale_cam_surface(surf, target_w: int, target_h: int):
    import pygame
    if surf.get_size() == (target_w, target_h):
        return surf
    return pygame.transform.smoothscale(surf, (target_w, target_h))


def draw_panel(screen, layout: WindowLayout, font, sm_font, recording, ep_idx, saved,
               step, reward, done, info, gripper_open, camera_name: str):
    """Draw status panel on the right side of the window."""
    px = layout.cam_w + 6

    def text(msg, y, color=WHITE, f=None):
        f = f or font
        screen.blit(f.render(msg, True, color), (px, y))

    # recording indicator
    rec_color = RED if recording else GRAY
    rec_label  = "● REC" if recording else "○ REC"
    text(rec_label, 10, rec_color)

    text(f"episode: {ep_idx}", 40)
    text(f"saved:   {saved}",  62)
    text(f"step:    {step}",   84)
    text(f"reward:  {reward:.3f}", 106)
    text(f"view:    {camera_name}", 128, GRAY, sm_font)

    # outcome
    if done:
        if info.get("success"):          outcome, col = "SUCCESS", GREEN
        elif info.get("arm_collision"):  outcome, col = "COLLISION", RED
        elif info.get("out_of_bounds"):  outcome, col = "OUT-OF-BOUNDS", YELLOW
        elif info.get("box_fell"):       outcome, col = "BOX FELL", RED
        else:                            outcome, col = "TIMEOUT", GRAY
        text(outcome, 150, col)

    # gripper state
    g_label = "grip: OPEN" if gripper_open else "grip: CLOSE"
    text(g_label, 176, GRAY, sm_font)

    # key guide
    guide = [
        "",
        "W/S  back / fwd (X)",
        "A/D  left / right (Y)",
        "Q/E  up / down",
        "Z/C  grip yaw",
        "SPC  gripper",
        "LMB  rotate  RMB pan",
        "wheel +/- zoom",
        "F11  fullscreen",
        "Tab / 1-6 view",
        "",
        "Enter save + reset",
        "N    discard + reset",
        "ESC  quit",
    ]
    for i, line in enumerate(guide):
        text(line, 200 + i * 18, GRAY, sm_font)


# ── main loop ─────────────────────────────────────────────────────────────────

def _dbg(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[teleop dbg] {msg}", flush=True)


def _release_egl_context(dbg: bool) -> None:
    """Release NVIDIA EGL context so SDL/pygame can use GLX on the same X display."""
    try:
        from robosuite.renderers.context import egl_context as ec
        ok = ec.EGL.eglMakeCurrent(
            ec.EGL_DISPLAY,
            ec.EGL.EGL_NO_SURFACE,
            ec.EGL.EGL_NO_SURFACE,
            ec.EGL.EGL_NO_CONTEXT,
        )
        _dbg(dbg, f"eglMakeCurrent(NO_CONTEXT) -> {ok}")
    except Exception as e:
        _dbg(dbg, f"release EGL failed: {e}")


def _bind_egl_for_render(env, dbg: bool) -> None:
    try:
        env.sim._render_context_offscreen.gl_ctx.make_current()
    except Exception as e:
        _dbg(dbg, f"bind EGL for render failed: {e}")


def _print_gl_env(enabled: bool, label: str) -> None:
    if not enabled:
        return
    keys = (
        "DISPLAY", "WAYLAND_DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID", "SDL_VIDEO_X11_FORCE_EGL", "SDL_VIDEODRIVER",
        "LIBGL_ALWAYS_SOFTWARE", "CUDA_VISIBLE_DEVICES",
    )
    print(f"[teleop dbg] === {label} ===", flush=True)
    for k in keys:
        print(f"[teleop dbg]   {k}={os.environ.get(k, '<unset>')!r}", flush=True)
    try:
        import robosuite.utils.binding_utils as bu
        print(f"[teleop dbg]   binding_utils._MUJOCO_GL={bu._MUJOCO_GL!r}", flush=True)
    except Exception as e:
        print(f"[teleop dbg]   binding_utils: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demos", help="output directory for LeRobot dataset")
    ap.add_argument("--repo-id", default="pushbox/demos",
                    help="LeRobot dataset repo_id (e.g. pushbox/demos)")
    ap.add_argument("--horizon", type=int, default=2000,
                    help="max steps per episode (timeout)")
    ap.add_argument("--debug", action="store_true", help="print GL/DISPLAY diagnostics")
    args = ap.parse_args()
    dbg = args.debug

    if not HAS_LEROBOT:
        print("[ERROR] pushbox.lerobot_io not importable; cannot write dataset", file=sys.stderr)
        sys.exit(1)

    _print_gl_env(dbg, "startup (before any heavy import)")

    out_dir = Path(args.out)

    # ── LeRobot dataset ──
    features = _get_lerobot_features()
    writer = LeRobotWriter.create(
        repo_id=args.repo_id,
        fps=CONTROL_FREQ,
        features=features,
        use_videos=True,
        root=str(out_dir),
    )
    ep_idx = 0

    # Hide DISPLAY so mujoco uses pure EGL (no X11/GLX probing at all).
    _display = os.environ.pop("DISPLAY", None)
    _dbg(dbg, f"popped DISPLAY={_display!r}")

    _print_gl_env(dbg, "before PushBoxEnv import")
    _dbg(dbg, "importing PushBoxEnv (loads robosuite/mujoco)...")
    from envs import PushBoxEnv
    _print_gl_env(dbg, "after PushBoxEnv import")

    _dbg(dbg, "creating PushBoxEnv (EGL offscreen)...")
    env = PushBoxEnv(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=args.horizon,
        hard_reset=False,  # hard_reset recreates EGL ctx; breaks after pygame window
        camera_names="top45,sideview",
        camera_heights=CAM_H,
        camera_widths=CAM_W,
    )
    _dbg(dbg, "PushBoxEnv created OK")

    if _display is not None:
        os.environ["DISPLAY"] = _display
    _dbg(dbg, f"restored DISPLAY={os.environ.get('DISPLAY')!r}")
    _print_gl_env(dbg, "before pygame.init")

    _release_egl_context(dbg)

    import pygame
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "50,50")
    _dbg(dbg, "calling pygame.init()...")
    pygame.init()
    _dbg(dbg, f"pygame video driver: {pygame.display.get_driver()!r}")
    layout = WindowLayout()
    layout.refresh()
    screen = layout.set_mode()
    _dbg(dbg, "pygame set_mode OK")
    pygame.display.set_caption("PushBox Teleop")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 15, bold=True)
    sm_font= pygame.font.SysFont("monospace", 13)

    # ── state ──
    _dbg(dbg, "calling env.reset()...")
    obs      = reset_env(env)
    _dbg(dbg, "env.reset() OK")
    buf      = EpisodeFrameBuffer()
    saved    = 0
    gripper_open = False           # gripper is closed after auto-grip
    grip_close_pending = False
    grip_close_counter = 0
    lift_phase = False
    eef_z_at_close: float | None = None
    done     = False
    reward   = 0.0
    info: dict = {}
    step     = 0
    view_idx = 0          # 0 = orbit, 1..4 = fixed presets
    orbit    = OrbitCamera()
    cam_drag = False
    cam_pan = False
    cam_last = (0, 0)

    def view_label() -> str:
        return VIEW_CYCLE[view_idx]

    def in_cam_panel(pos) -> bool:
        return pos[0] < layout.cam_w

    running = True
    while running:
        # ── events ──
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False

                elif ev.key == pygame.K_SPACE:
                    if gripper_open:
                        gripper_open = False
                        grip_close_pending = True
                        grip_close_counter = 0
                        lift_phase = False
                        eef_z_at_close = None
                    else:
                        gripper_open = True
                        grip_close_pending = False
                        grip_close_counter = 0
                        lift_phase = False
                        eef_z_at_close = None

                elif ev.key == pygame.K_RETURN:
                    if len(buf) > 0:
                        save_episode_lerobot(writer, buf, ep_idx)
                        ep_idx += 1
                        saved  += 1
                    obs = reset_env(env)
                    gripper_open = False
                    grip_close_pending = False
                    grip_close_counter = 0
                    lift_phase = False
                    eef_z_at_close = None
                    done = False; step = 0; reward = 0.0; info = {}

                elif ev.key == pygame.K_n:
                    print(f"[discard] episode discarded ({len(buf)} steps)")
                    buf.clear()
                    obs = reset_env(env)
                    gripper_open = False
                    grip_close_pending = False
                    grip_close_counter = 0
                    lift_phase = False
                    eef_z_at_close = None
                    done = False; step = 0; reward = 0.0; info = {}

                elif ev.key == pygame.K_TAB:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        view_idx = (view_idx - 1) % len(VIEW_CYCLE)
                    else:
                        view_idx = (view_idx + 1) % len(VIEW_CYCLE)

                elif pygame.K_1 <= ev.key <= pygame.K_4:
                    view_idx = (ev.key - pygame.K_1) + 1
                elif ev.key == pygame.K_5:
                    view_idx = VIEW_CYCLE.index("top45")
                elif ev.key == pygame.K_6:
                    view_idx = VIEW_CYCLE.index("sideview")  # robosuite built-in sideview

                elif ev.key == pygame.K_0:
                    view_idx = 0  # orbit

                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    view_idx = 0
                    orbit.zoom_wheel(1)

                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    view_idx = 0
                    orbit.zoom_wheel(-1)

                elif ev.key == pygame.K_F11:
                    layout.fullscreen = not layout.fullscreen
                    layout.refresh()
                    screen = layout.set_mode()

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if in_cam_panel(ev.pos):
                    if ev.button == 1:
                        view_idx = 0
                        cam_drag = True
                        cam_pan = False
                        cam_last = ev.pos
                    elif ev.button == 3:
                        view_idx = 0
                        cam_pan = True
                        cam_drag = False
                        cam_last = ev.pos
                    elif ev.button in (4, 5):  # scroll-as-button (Linux)
                        view_idx = 0
                        orbit.zoom_wheel(1 if ev.button == 4 else -1)

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    cam_drag = False
                elif ev.button == 3:
                    cam_pan = False

            elif ev.type == pygame.MOUSEMOTION:
                if in_cam_panel(ev.pos) and (cam_drag or cam_pan):
                    dx = ev.pos[0] - cam_last[0]
                    dy = ev.pos[1] - cam_last[1]
                    cam_last = ev.pos
                    if dx or dy:
                        if cam_drag:
                            orbit.rotate(dx, dy)
                        elif cam_pan:
                            orbit.pan(dx, dy)

            elif ev.type == pygame.MOUSEWHEEL:
                if in_cam_panel(pygame.mouse.get_pos()):
                    view_idx = 0
                    orbit.zoom_wheel(ev.y)

        # ── step ──
        if not done:
            pressed = pygame.key.get_pressed()

            if lift_phase:
                action = np.array([0.0, 0.0, LIFT_DZ, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            elif grip_close_pending:
                grip_close_counter += 1
                if grip_close_counter >= CLOSE_DURATION:
                    lift_phase = True
                    eef_z_at_close = float(obs["robot0_eef_pos"][2])
                    action = np.array([0.0, 0.0, LIFT_DZ, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    action = keys_to_action(pressed, gripper_open)
            else:
                action = keys_to_action(pressed, gripper_open)

            obs, reward, done, info = env.step(action)
            step += 1

            if lift_phase:
                current_z = float(obs["robot0_eef_pos"][2])
                if eef_z_at_close is not None and current_z - eef_z_at_close >= 0.03:
                    grip_close_pending = False
                    grip_close_counter = 0
                    lift_phase = False
                    eef_z_at_close = None

            rec_obs = dict(obs)
            top45 = capture_cam(env, "top45")
            sideview = capture_cam(env, "sideview")
            joints = obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32))
            gripper_q = obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
            agent_state = np.concatenate([joints, gripper_q]).astype(np.float32)
            box_pos = obs.get("box_pos_xy", np.zeros(2, dtype=np.float32)).astype(np.float32)

            lerobot_frame = {
                "observation.images.top45": top45,
                "observation.images.sideview": sideview,
                "observation.state": agent_state,
                "observation.box_pos": box_pos,
                "action": action,
                "task": "push_box",
            }
            buf.append(lerobot_frame)

        # ── render ──
        screen.fill(BG)

        label = view_label()
        _bind_egl_for_render(env, dbg)
        if view_idx == 0:
            cam_surf = render_frame_orbit(env, orbit)
        else:
            cam_surf = render_frame_fixed(env, label)
        cam_surf = _scale_cam_surface(cam_surf, layout.cam_w, layout.cam_h)
        screen.blit(cam_surf, (0, 0))

        draw_panel(screen, layout, font, sm_font, True, ep_idx, saved,
                   step, reward, done, info, gripper_open, label)

        pygame.display.flip()
        clock.tick(20)   # ~20 fps matches control_freq

    writer.finalize()
    env.close()
    pygame.quit()
    print(f"[done] {saved} episodes saved to {out_dir}/")


if __name__ == "__main__":
    main()
