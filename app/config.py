from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:admin@localhost:5432/trafficrl"
    sync_database_url: str = "postgresql+asyncpg://postgres:admin@localhost:5432/trafficrl"
    log_level: str = "INFO"

    # DQN defaults — overridable via request body
    state_dim: int = 4       # [N, S, E, W] queue lengths
    action_dim: int = 2      # 0 = NS green, 1 = EW green

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
