"""PushBox task paths (isolated from other ARP tasks)."""
from pathlib import Path

# LeRobot v3 dataset directory (contains meta/, data/, videos/)
DEFAULT_DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "datagen"

EXPECTED_NUM_EPISODES = 101
NUM_TRAIN_EPISODES = 80
NUM_VAL_EPISODES = 21
VAL_RATIO = NUM_VAL_EPISODES / EXPECTED_NUM_EPISODES  # ~0.208
