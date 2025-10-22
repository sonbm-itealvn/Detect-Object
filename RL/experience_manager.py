# File: experience_manager.py
import os
import datetime
from collections import deque
from typing import Deque, Dict, List, Any, Optional

import torch


class ExperienceManager:
    """Manage replay buffer persistence across training epochs."""

    def __init__(
        self,
        base_dir: str = "experiences",
        main_filename: str = "main_replay_buffer.pt",
        max_length: int = 50000,
    ) -> None:
        self.base_dir = base_dir
        self.epoch_dir = os.path.join(self.base_dir, "epoch_buffers")
        self.main_path = os.path.join(self.base_dir, main_filename)
        self.max_length = max_length

        os.makedirs(self.epoch_dir, exist_ok=True)

        self.memory: Deque[Dict[str, Any]] = deque(maxlen=self.max_length)
        self._load_main_buffer()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def add_batch(self, experiences: List[Dict[str, Any]], auto_save: bool = True) -> None:
        """Append a batch of experiences to the in-memory buffer."""
        if not experiences:
            return

        for exp in experiences:
            self.memory.append(self._sanitize_experience(exp))

        if auto_save:
            self._save_main_buffer()

    def record_epoch_batch(
        self,
        experiences: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Persist per-epoch experiences and merge them into the main buffer."""
        if not experiences:
            return None

        sanitized = [self._sanitize_experience(exp) for exp in experiences]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        exp_id = metadata.get("experiment_id") if metadata else None
        filename = f"{exp_id or 'default_run'}_experiences.pt"

        filepath = os.path.join(self.epoch_dir, filename)
        existing_experiences: List[Dict[str, Any]] = []
        existing_metadata: Dict[str, Any] = {}

        if os.path.exists(filepath):
            try:
                existing_payload = torch.load(filepath, map_location="cpu")
                existing_experiences = existing_payload.get("experiences", [])
                existing_metadata = existing_payload.get("metadata", {})
            except Exception:
                existing_experiences = []
                existing_metadata = {}

        combined_experiences = existing_experiences + sanitized
        combined_metadata = self._merge_epoch_metadata(
            existing_metadata,
            metadata or {},
            total_experiences=len(combined_experiences),
            timestamp=timestamp,
        )

        payload = {
            "metadata": combined_metadata,
            "experiences": combined_experiences,
            "saved_at": timestamp,
        }

        torch.save(payload, filepath)
        self.add_batch(sanitized, auto_save=True)
        return filepath

    def merge_epoch_file(self, filepath: str, auto_save: bool = True) -> bool:
        """Load an epoch buffer from disk and merge into the main buffer."""
        if not os.path.exists(filepath):
            return False

        try:
            payload = torch.load(filepath, map_location="cpu")
        except Exception:
            return False

        experiences = payload.get("experiences", [])
        if not experiences:
            return False

        for exp in experiences:
            self.memory.append(self._sanitize_experience(exp))

        if auto_save:
            self._save_main_buffer()
        return True

    def save(self) -> None:
        """Force-save the main replay buffer."""
        self._save_main_buffer()

    def load(self) -> None:
        """Reload the main replay buffer from disk."""
        self._load_main_buffer()

    def clear(self, keep_files: bool = True) -> None:
        """Clear in-memory buffer; optionally remove persisted files."""
        self.memory.clear()
        if not keep_files:
            if os.path.exists(self.main_path):
                os.remove(self.main_path)
            for file in os.listdir(self.epoch_dir):
                os.remove(os.path.join(self.epoch_dir, file))

    def stats(self) -> Dict[str, Any]:
        """Return basic statistics about the replay buffer."""
        return {
            "buffer_size": len(self.memory),
            "capacity": self.max_length,
            "is_full": len(self.memory) >= self.max_length,
            "main_path": self.main_path,
            "epoch_dir": self.epoch_dir,
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_main_buffer(self) -> None:
        if not os.path.exists(self.main_path):
            return

        try:
            saved = torch.load(self.main_path, map_location="cpu")
            experiences = saved.get("experiences", [])
            for exp in experiences:
                self.memory.append(self._sanitize_experience(exp))
        except Exception:
            # If loading fails we start with an empty buffer but keep the file for debugging.
            self.memory.clear()

    def _save_main_buffer(self) -> None:
        payload = {
            "saved_at": datetime.datetime.now().isoformat(),
            "experiences": list(self.memory),
        }
        torch.save(payload, self.main_path)

    def _sanitize_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-serializable payload such as PIL images."""
        sanitized = {}
        for key, value in experience.items():
            if key == "metadata" and isinstance(value, dict):
                sanitized[key] = self._strip_image_keys(value)
            elif isinstance(value, dict):
                sanitized[key] = self._strip_image_keys(value)
            elif isinstance(value, list):
                sanitized[key] = [self._strip_image_keys(item) if isinstance(item, dict) else item for item in value]
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def _strip_image_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Drop keys that could contain binary image data."""
        blacklist = {"image", "image_data", "pixel_values"}
        return {k: ExperienceManager._strip_image_keys(v) if isinstance(v, dict) else v
                for k, v in payload.items() if k not in blacklist}

    def _merge_epoch_metadata(
        self,
        existing: Dict[str, Any],
        new: Dict[str, Any],
        total_experiences: int,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Merge metadata across epochs into a single experiment summary."""
        merged: Dict[str, Any] = dict(existing or {})

        if new.get("experiment_id"):
            merged["experiment_id"] = new["experiment_id"]
        if new.get("experiment_dir"):
            merged["experiment_dir"] = new["experiment_dir"]

        history: List[Dict[str, Any]] = []
        if isinstance(merged.get("epoch_history"), list):
            history = list(merged["epoch_history"])

        epoch_entry = {
            key: new.get(key)
            for key in ("epoch", "reward", "detection_loss", "relationship_loss", "timestamp")
            if new.get(key) is not None
        }
        if epoch_entry:
            epoch_value = epoch_entry.get("epoch")
            if epoch_value is not None:
                history = [item for item in history if item.get("epoch") != epoch_value]
            history.append(epoch_entry)
            history.sort(key=lambda item: item.get("epoch", float("inf")))
            merged["latest_epoch"] = epoch_value
            if "reward" in epoch_entry:
                merged["last_reward"] = epoch_entry["reward"]

        merged["epoch_history"] = history
        merged["total_epochs"] = len(history)
        merged["total_experiences"] = total_experiences
        merged["last_updated"] = timestamp

        return merged

