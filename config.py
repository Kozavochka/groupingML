import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    model_checkpoint_path: str = os.getenv("MODEL_CHECKPOINT_PATH", "artifacts/ckpt_best.pt")
    model_emb_dim: int = int(os.getenv("MODEL_EMB_DIM", "16"))
    model_device: str = os.getenv("MODEL_DEVICE", "").strip()

    api_auth_enabled: bool = _as_bool(os.getenv("API_AUTH_ENABLED"), True)
    api_auth_username: str = os.getenv("API_AUTH_USERNAME", "admin")
    api_auth_password: str = os.getenv("API_AUTH_PASSWORD", "change_me")

    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


settings = Settings()

