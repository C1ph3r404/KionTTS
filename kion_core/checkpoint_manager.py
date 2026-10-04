"""
Production Checkpoint Manager for KionTTS.
Handles state serialization, auto-resumption, local rotation, and automated
Hugging Face Hub synchronization for 12-hour Kaggle/Colab sessions.
"""

import os
import glob
import re
import shutil
import torch
from typing import Dict, Any, Optional, List


def get_hf_token() -> Optional[str]:
    """Discovers Hugging Face token across Kaggle secrets, Colab secrets, env vars, or HF cache."""
    # 1. Kaggle Secrets (Primary)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for sec_name in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token", "HF_API_TOKEN"]:
            try:
                t = user_secrets.get_secret(sec_name)
                if t and len(t.strip()) > 0:
                    print(f"[✓] Retrieved Hugging Face token from Kaggle Secret: '{sec_name}'")
                    return t.strip()
            except Exception:
                pass
    except Exception:
        pass

    # 2. Colab Secrets
    try:
        from google.colab import userdata
        for sec_name in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token"]:
            try:
                t = userdata.get(sec_name)
                if t and len(t.strip()) > 0:
                    return t.strip()
            except Exception:
                pass
    except Exception:
        pass

    # 3. Environment Variables
    for env_var in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_AUTH_TOKEN", "hf_token"]:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    # 4. Hugging Face Cache
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached.strip()
    except Exception:
        pass

    return None


class KionCheckpointManager:
    """
    Manages saving, loading, rolling pruning, and Hugging Face Hub sync of KionTTS checkpoints.
    """
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        keep_last_n: int = 3,
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.repo_id = repo_id or os.environ.get("HF_REPO_ID", "nate0001/KionTTS")
        self.hf_token = hf_token or get_hf_token()
        self.keep_last_n = keep_last_n

        self.api = None
        if self.hf_token:
            try:
                from huggingface_hub import HfApi
                self.api = HfApi(token=self.hf_token)
                try:
                    self.api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True)
                    print(f"[✓] Connected to Hugging Face Hub repository: {self.repo_id}")
                except Exception as e:
                    print(f"[-] Hugging Face Hub notice: {e}")
            except Exception as e:
                print(f"[-] Hugging Face Hub library not available or failed to initialize: {e}")
        else:
            print("[*] HF_TOKEN not detected. Checkpoints will be saved locally.")

    def save_checkpoint(
        self,
        stage: str,
        epoch: int,
        step: int,
        models: Dict[str, Any],
        optimizers: Optional[Dict[str, Any]] = None,
        loss_val: Optional[float] = None,
        is_best: bool = False,
        upload_hf: bool = True,
    ) -> str:
        """
        Saves a comprehensive training checkpoint containing all neural submodules,
        optimizers, step counters, and validation metrics.
        """
        net_states = {}
        for k, v in models.items():
            if v is None:
                continue
            if hasattr(v, "module"):  # Handle DDP wrapped modules
                net_states[k] = v.module.state_dict()
            elif hasattr(v, "state_dict"):
                net_states[k] = v.state_dict()

        opt_states = {}
        if optimizers:
            for k, v in optimizers.items():
                if v is not None and hasattr(v, "state_dict"):
                    opt_states[k] = v.state_dict()

        payload = {
            "stage": stage,
            "epoch": epoch,
            "step": step,
            "loss_val": loss_val,
            "net": net_states,
            "optimizers": opt_states,
        }

        # Save step/epoch file
        step_filename = f"kion_{stage}_epoch_{epoch:03d}_step_{step:06d}.pth"
        step_path = os.path.join(self.checkpoint_dir, step_filename)
        torch.save(payload, step_path)
        print(f"[✓] Saved local checkpoint: {step_path} ({os.path.getsize(step_path) / (1024*1024):.1f} MB)")

        # Update latest pointer & file
        latest_filename = f"kion_{stage}_latest.pth"
        latest_path = os.path.join(self.checkpoint_dir, latest_filename)
        shutil.copy2(step_path, latest_path)

        pointer_path = os.path.join(self.checkpoint_dir, f"latest_{stage}_checkpoint.txt")
        with open(pointer_path, "w") as f:
            f.write(step_filename)

        if is_best:
            best_filename = f"kion_{stage}_best.pth"
            best_path = os.path.join(self.checkpoint_dir, best_filename)
            shutil.copy2(step_path, best_path)
            print(f"[★] New best validation checkpoint saved: {best_path}")

        # Local pruning
        self.prune_local_checkpoints(stage=stage)

        # Remote sync to Hugging Face Hub
        if upload_hf and self.api:
            self._upload_file(step_path, step_filename)
            self._upload_file(latest_path, latest_filename)
            self._upload_file(pointer_path, f"latest_{stage}_checkpoint.txt")
            if is_best:
                self._upload_file(best_path, f"kion_{stage}_best.pth")
            self.prune_hf_checkpoints(stage=stage)

        return step_path

    def _upload_file(self, local_path: str, repo_filename: str):
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_filename,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Sync checkpoint: {repo_filename}",
            )
            print(f"[✓] Synced {repo_filename} to Hugging Face Hub.")
        except Exception as e:
            print(f"[!] Warning: Failed uploading {repo_filename} to HF: {e}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        models: Dict[str, Any],
        optimizers: Optional[Dict[str, Any]] = None,
        load_optimizers: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads weights into model submodules and restores optimizer states.
        """
        print(f"[*] Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        net = state.get("net", state)

        for k, mod in models.items():
            if mod is None:
                continue
            if k in net:
                target_mod = mod.module if hasattr(mod, "module") else mod
                try:
                    target_mod.load_state_dict(net[k], strict=False)
                    print(f"  [✓] Loaded weights for: {k}")
                except Exception as e:
                    print(f"  [!] Warning loading {k}: {e}")
            else:
                print(f"  [-] Key '{k}' not found in checkpoint state_dict.")

        if load_optimizers and optimizers and "optimizers" in state:
            opt_dict = state["optimizers"]
            for k, opt in optimizers.items():
                if k in opt_dict and opt is not None:
                    try:
                        opt.load_state_dict(opt_dict[k])
                        print(f"  [✓] Restored optimizer: {k}")
                    except Exception as e:
                        print(f"  [!] Could not restore optimizer {k}: {e}")

        return {
            "stage": state.get("stage", "stage1"),
            "epoch": state.get("epoch", 0),
            "step": state.get("step", 0),
            "loss_val": state.get("loss_val", None),
        }

    def find_latest_checkpoint(self, stage: str = "stage1") -> Optional[str]:
        """Discovers the most recent checkpoint locally or on Hugging Face Hub."""
        # 1. Local pointer
        ptr = os.path.join(self.checkpoint_dir, f"latest_{stage}_checkpoint.txt")
        if os.path.exists(ptr):
            with open(ptr, "r") as f:
                target = f.readline().strip()
            target_path = os.path.join(self.checkpoint_dir, target)
            if os.path.exists(target_path):
                return target_path

        # 2. Local glob
        candidates = glob.glob(os.path.join(self.checkpoint_dir, f"kion_{stage}_epoch_*.pth"))
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            return candidates[0]

        # 3. Fallback names
        for fallback in [f"kion_{stage}_best.pth", f"kion_{stage}_latest.pth"]:
            cand = os.path.join(self.checkpoint_dir, fallback)
            if os.path.exists(cand):
                return cand

        # 4. Hugging Face Hub download
        if self.api:
            try:
                from huggingface_hub import hf_hub_download
                # Try downloading pointer
                try:
                    down_ptr = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=f"latest_{stage}_checkpoint.txt",
                        local_dir=self.checkpoint_dir,
                    )
                    with open(down_ptr, "r") as f:
                        remote_target = f.readline().strip()
                    if remote_target:
                        print(f"[*] Downloading {remote_target} from HF Hub...")
                        return hf_hub_download(
                            repo_id=self.repo_id,
                            filename=remote_target,
                            local_dir=self.checkpoint_dir,
                        )
                except Exception:
                    pass

                # Scan repo files directly for highest step number
                try:
                    files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="model")
                    pattern = re.compile(rf"kion_{stage}_epoch_(\d+)_step_(\d+)\.pth")
                    matched_files = []
                    for f in files:
                        m = pattern.search(f)
                        if m:
                            ep, st = int(m.group(1)), int(m.group(2))
                            matched_files.append((st, ep, f))
                    if matched_files:
                        matched_files.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        latest_remote = matched_files[0][2]
                        print(f"[*] Downloading latest step checkpoint from HF Hub: {latest_remote}")
                        return hf_hub_download(
                            repo_id=self.repo_id,
                            filename=latest_remote,
                            local_dir=self.checkpoint_dir,
                        )
                except Exception as e:
                    print(f"[-] Remote file list check: {e}")

                # Try standard fallback names on HF
                for fallback in [f"kion_{stage}_latest.pth", f"kion_{stage}_best.pth"]:
                    try:
                        return hf_hub_download(
                            repo_id=self.repo_id,
                            filename=fallback,
                            local_dir=self.checkpoint_dir,
                        )
                    except Exception:
                        pass
            except Exception as e:
                print(f"[-] Could not query Hugging Face Hub: {e}")

        return None

    def prune_local_checkpoints(self, stage: str):
        """Keeps only the latest N step checkpoints on local disk to prevent filling storage."""
        candidates = glob.glob(os.path.join(self.checkpoint_dir, f"kion_{stage}_epoch_*.pth"))
        if len(candidates) > self.keep_last_n:
            candidates.sort(key=os.path.getmtime)
            to_delete = candidates[:-self.keep_last_n]
            for path in to_delete:
                try:
                    os.remove(path)
                    print(f"[-] Pruned old local checkpoint: {os.path.basename(path)}")
                except Exception:
                    pass

    def prune_hf_checkpoints(self, stage: str):
        """Keeps only the latest N step checkpoints on Hugging Face Hub."""
        if not self.api:
            return
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="model")
            step_files = []
            pattern = re.compile(rf"kion_{stage}_epoch_(\d+)_step_(\d+)\.pth")
            for f in files:
                m = pattern.match(f)
                if m:
                    step_num = int(m.group(2))
                    step_files.append((step_num, f))
            step_files.sort(key=lambda x: x[0])
            if len(step_files) > self.keep_last_n:
                to_delete = step_files[:-self.keep_last_n]
                for _, fname in to_delete:
                    try:
                        self.api.delete_file(
                            path_in_repo=fname,
                            repo_id=self.repo_id,
                            repo_type="model",
                            commit_message=f"Prune old step checkpoint: {fname}",
                        )
                        print(f"[-] Pruned old HF checkpoint: {fname}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"[-] HF pruning check skipped: {e}")
