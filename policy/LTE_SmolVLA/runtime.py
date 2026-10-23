"""Use the dependency installed by XPolicyLab's SmolVLA adapter explicitly."""

import sys
from pathlib import Path


def use_xpolicy_smolvla():
    source = Path(__file__).resolve().parents[1] / "SmolVLA" / "smovla" / "src"
    if not (source / "lerobot" / "policies" / "smolvla").is_dir():
        raise ImportError("Install policy/SmolVLA first: bash policy/LTE_SmolVLA/install.sh")
    loaded = sys.modules.get("lerobot")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
        raise ImportError(f"Another LeRobot is already imported from {loaded.__file__}; use a fresh SmolVLA process")
    sys.path.insert(0, str(source))
    return source
