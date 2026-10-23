#!/usr/bin/env python3
"""Rebuild deploy-time processor files for a model-only LTE-SmolVLA artifact.

This recovery utility creates a new, evaluation-only ``pretrained_model``
directory. It never modifies the source artifact or an existing checkpoint.
"""

import argparse
import json
import shutil
from dataclasses import fields
from pathlib import Path

import wandb
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import RenameObservationsProcessorStep
from lerobot.policies.factory import make_pre_post_processors

from .configuration_lte_smolvla import LTESmolVLAConfig
from .train import CAMERA_RENAME, configure_history


REQUIRED_MODEL_FILE = "model.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-run", required=True, help="entity/project/run-id")
    parser.add_argument("--artifact", required=True, help="Fully qualified W&B model artifact")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    run = wandb.Api().run(args.wandb_run)
    policy_config = run.config.get("policy")
    if not isinstance(policy_config, dict):
        raise ValueError("The W&B run does not contain a policy configuration")
    if policy_config.get("type") != "lte_smolvla":
        raise ValueError(f"Expected lte_smolvla policy, got {policy_config.get('type')!r}")

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    config_fields = {field.name for field in fields(LTESmolVLAConfig) if field.init}
    config = LTESmolVLAConfig(**{key: value for key, value in policy_config.items() if key in config_fields})
    configure_history(config, dataset, CAMERA_RENAME)

    args.output.mkdir(parents=True)
    artifact_dir = Path(wandb.Api().artifact(args.artifact).download(root=str(args.output / "wandb-artifact")))
    model_file = artifact_dir / REQUIRED_MODEL_FILE
    if not model_file.is_file():
        raise FileNotFoundError(f"Artifact does not contain {REQUIRED_MODEL_FILE}: {artifact_dir}")

    config.save_pretrained(args.output)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    for step in preprocessor.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            step.rename_map = dict(CAMERA_RENAME)
    preprocessor.save_pretrained(args.output)
    postprocessor.save_pretrained(args.output)
    shutil.copy2(model_file, args.output / REQUIRED_MODEL_FILE)

    provenance = {
        "recovery_type": "processor-reconstruction",
        "wandb_run": args.wandb_run,
        "artifact": args.artifact,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "camera_rename": CAMERA_RENAME,
        "note": "Processors were regenerated from the dataset metadata statistics; the model weights were copied unchanged from the W&B artifact.",
    }
    (args.output / "recovery_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
