"""
Colab Cell 02: Drive Checkpoint Manager
Provides persistent checkpoint saving to Google Drive and automatic resumption
from the latest available checkpoint upon runtime restart.
"""

import os
import glob
import re
import shutil
import torch
from typing import Optional, Dict, Any, Tuple


class DriveCheckpointManager:
    """
    Manages saving and loading model checkpoints to/from Google Drive.
    Ensures safe atomic writes to avoid data corruption if Colab disconnects.
    """

    def __init__(
        self,
        checkpoint_dir: str = "/content/drive/MyDrive/KionTTS_Checkpoints",
        keep_last_n: int = 3,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_symlink = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Scans the checkpoint directory and returns the path to the newest checkpoint.
        Checks for latest_checkpoint.pt or highest step numbered checkpoint.
        """
        # First check explicit latest file
        if os.path.exists(self.latest_symlink):
            return self.latest_symlink

        # Search for numbered checkpoints
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None

        # Sort by step number extracted from filename
        def _get_step(path):
            match = re.search(r"checkpoint_step_(\d+)\.pt", path)
            return int(match.group(1)) if match else -1

        checkpoints.sort(key=_get_step)
        return checkpoints[-1]

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        step: int,
        epoch: int,
        val_loss: Optional[float] = None,
        is_best: bool = False,
    ) -> str:
        """
        Saves checkpoint state atomically to Drive.
        """
        checkpoint_filename = f"checkpoint_step_{step:07d}.pt"
        target_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        temp_path = os.path.join(self.checkpoint_dir, f".tmp_{checkpoint_filename}")

        state["step"] = step
        state["epoch"] = epoch
        state["val_loss"] = val_loss

        # Write to temporary file first, then atomic rename
        torch.save(state, temp_path)
        shutil.move(temp_path, target_path)

        # Copy to latest_checkpoint.pt
        shutil.copyfile(target_path, self.latest_symlink)

        if is_best:
            shutil.copyfile(target_path, self.best_checkpoint_path)
            print(f"[*] New best model saved to {self.best_checkpoint_path} (Val Loss: {val_loss:.4f})")

        print(f"[+] Checkpoint saved successfully: {target_path}")
        self._prune_old_checkpoints()
        return target_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Tuple[Optional[Dict[str, Any]], int, int, float]:
        """
        Loads state from checkpoint. If path is not provided, automatically
        discovers and loads the latest checkpoint.
        Returns: (state_dict, start_step, start_epoch, best_val_loss)
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("[-] No existing checkpoint found on Drive. Starting from scratch (Epoch 0, Step 0).")
            return None, 0, 0, float("inf")

        print(f"[*] Resuming training from Drive checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        step = state.get("step", 0)
        epoch = state.get("epoch", 0)
        best_val_loss = state.get("best_val_loss", float("inf"))

        print(f"[*] Checkpoint loaded: Resuming at Epoch {epoch}, Global Step {step}, Best Val Loss {best_val_loss:.4f}")
        return state, step, epoch, best_val_loss

    def _prune_old_checkpoints(self):
        """Keep only the latest N numbered checkpoints to preserve Google Drive storage."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        checkpoints = glob.glob(pattern)

        def _get_step(path):
            match = re.search(r"checkpoint_step_(\d+)\.pt", path)
            return int(match.group(1)) if match else -1

        checkpoints.sort(key=_get_step)
        if len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[: -self.keep_last_n]
            for ckpt in to_remove:
                try:
                    os.remove(ckpt)
                    print(f"[-] Pruned old checkpoint: {os.path.basename(ckpt)}")
                except OSError as e:
                    print(f"Warning: Could not remove {ckpt}: {e}")


if __name__ == "__main__":
    # Test Checkpoint Manager logic locally
    test_dir = "/tmp/kion_test_checkpoints"
    manager = DriveCheckpointManager(checkpoint_dir=test_dir, keep_last_n=2)
    dummy_state = {"model_state": {}}
    manager.save_checkpoint(dummy_state, step=100, epoch=1, val_loss=1.5)
    manager.save_checkpoint(dummy_state, step=200, epoch=2, val_loss=1.2, is_best=True)
    manager.save_checkpoint(dummy_state, step=300, epoch=3, val_loss=1.4)

    latest = manager.find_latest_checkpoint()
    print("Latest checkpoint detected:", latest)
    loaded_state, step, epoch, best_loss = manager.load_checkpoint()
    assert step == 300
    print("[Cell 02 Test] Checkpoint manager PASSED.")
    shutil.rmtree(test_dir)
