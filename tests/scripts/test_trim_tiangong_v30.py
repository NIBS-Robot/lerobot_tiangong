import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_tool():
    path = Path(__file__).parents[2] / "tools" / "trim_tiangong_v30.py"
    spec = importlib.util.spec_from_file_location("trim_tiangong_v30", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trim_bounds_removes_static_start_and_last_release_tail():
    tool = _load_tool()
    state = np.zeros((10, 34), dtype=np.float32)
    action = np.zeros((10, 20), dtype=np.float32)

    # Pose starts moving at frame 3, the right gripper closes at 4, and opens at 8.
    action[3:, 10] = 0.01
    action[4:8, 19] = 1.0

    trim = tool._trim_bounds(action, state, tolerance=1e-3)

    assert (trim.start, trim.end) == (3, 8)
    assert trim.has_release


def test_trim_bounds_keeps_tail_when_gripper_never_reopens():
    tool = _load_tool()
    state = np.zeros((6, 34), dtype=np.float32)
    action = np.zeros((6, 20), dtype=np.float32)

    action[2:, 10] = 0.01
    action[4:, 19] = 1.0

    trim = tool._trim_bounds(action, state, tolerance=1e-3)

    assert (trim.start, trim.end) == (2, 6)
    assert not trim.has_release
