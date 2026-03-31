from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Local fallback (used only if env not set)
    database_url: str = "postgresql+asyncpg://postgres:admin@localhost:5432/trafficrl"
    sync_database_url: str = "postgresql+asyncpg://postgres:admin@localhost:5432/trafficrl"
    log_level: str = "INFO"

    # DQN defaults
    state_dim: int = 4
    action_dim: int = 2

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()

    db_url = os.getenv("DATABASE_URL")

    if db_url:
        # 🔥 FORCE async driver (THIS FIXES YOUR ERROR)
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://")

        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

        settings.database_url = db_url
        settings.sync_database_url = db_url

    return settings