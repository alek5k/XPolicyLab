"""Use RobotWorkspace.run unchanged; add dataset binding and an update clock."""

from pathlib import Path
import json

from diffusion_policy.workspace.robotworkspace import RobotWorkspace


class LTEWorkspace(RobotWorkspace):
    include_keys = RobotWorkspace.include_keys + ["lte_optimizer_updates"]

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir)
        self.lte_optimizer_updates = 0
        self._optimizer_hook = self.optimizer.register_step_post_hook(self._after_optimizer_step)

    def _after_optimizer_step(self, optimizer, args, kwargs):
        self.lte_optimizer_updates += 1
        self.model.set_optimizer_update(self.lte_optimizer_updates)
        if self.ema_model is not None:
            self.ema_model.set_optimizer_update(self.lte_optimizer_updates)
        if self.lte_optimizer_updates == 1 or self.lte_optimizer_updates % self.cfg.lte.log_every_updates == 0:
            row = dict(self.model.last_lte_losses, completed_optimizer_updates=self.lte_optimizer_updates)
            log = Path(self.cfg.lte.checkpoint_dir) / "lte_metrics.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"LTE metrics: {json.dumps(row)}", flush=True)

    def save_checkpoint(self, path=None, **kwargs):
        # The inherited loop constructs a path under policy/DP. Keep its
        # epoch filename and native checkpoint payload, change only the root.
        root = Path(self.cfg.lte.checkpoint_dir)
        name = Path(path).name if path is not None else "latest.ckpt"
        return super().save_checkpoint(path=root / name, **kwargs)

    def load_payload(self, payload, **kwargs):
        super().load_payload(payload, **kwargs)
        self.model.set_optimizer_update(self.lte_optimizer_updates)
        if self.ema_model is not None:
            self.ema_model.set_optimizer_update(self.lte_optimizer_updates)

    def run(self):
        try:
            return super().run()
        finally:
            if self._saving_thread is not None:
                self._saving_thread.join()
