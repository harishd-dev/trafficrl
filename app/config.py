from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
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

    # 🔥 OVERRIDE for Render
    db_url = os.getenv("DATABASE_URL")

    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://")

        settings.database_url = db_url
        settings.sync_database_url = db_url

    return settings