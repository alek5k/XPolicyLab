"""Create tiny random SmolVLA assets and synthetic v3 data; no robot weights.

Uses an already-cached SmolVLM tokenizer/processor. Run with HF_HUB_OFFLINE=1.
Outputs are wiring fixtures only, never checkpoints for robot evaluation.
"""

import sys
from pathlib import Path

import numpy as np

from XPolicyLab.policy.LTE_SmolVLA.tests.test_lte_smolvla import tiny_vlm_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor


def main(root):
    assets = root / "vlm"
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)
    processor.save_pretrained(assets)
    tiny_vlm_config(len(processor.tokenizer)).save_pretrained(assets)
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        **{f"observation.images.{camera}": {"dtype": "image", "shape": (3, 16, 16), "names": ["channels", "height", "width"]}
           for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
    }
    dataset = LeRobotDataset.create("local/lte-smoke", fps=10, features=features, root=root / "dataset", use_videos=False)
    rng = np.random.default_rng(42)
    for _ in range(2):
        for _ in range(3):
            dataset.add_frame({
                "observation.state": rng.normal(size=14).astype(np.float32),
                "action": rng.normal(size=14).astype(np.float32),
                **{key: rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for key in features if key.startswith("observation.images.")},
                "task": "smoke test",
            })
        dataset.save_episode()
    dataset.finalize()
    print(root)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
