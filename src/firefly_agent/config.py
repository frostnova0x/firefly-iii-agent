"""Configuration loader.

Loads and validates two inputs:

1. Environment variables (loaded via os.environ) — runtime secrets
   (tokens, PATs, owner IDs). Sources, in order of precedence:
     - Docker: `env_file: - .env` in docker-compose.yml
     - systemd: EnvironmentFile= directive in the unit
     - bare: source the file manually before running
   The .env file should be `chmod 600` (owner-only).

2. `config.toml` (repo-relative) — non-secret behavioral config:
   categories, tags, model chains, UI flags.

Both are validated at startup. If anything is missing or malformed,
the process fails fast with a clear error — we never start the bot
in a half-configured state.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger(__name__)


# ============================================================
# Env-backed settings (secrets + runtime config)
# ============================================================


class EnvSettings(BaseModel):
    """Settings loaded from environment variables.

    Source: an .env file loaded by Docker compose, a systemd
    EnvironmentFile, or shell exports. The .env file must be
    chmod 600 (owner-only) — it contains every credential.
    """

    # --- Telegram ---
    telegram_bot_token: str = Field(..., min_length=20, description="BotFather token")
    telegram_owner_ids: list[int] = Field(..., min_length=1, description="Whitelisted Telegram user IDs")

    # --- OpenRouter ---
    openrouter_api_key: str = Field(..., min_length=20)
    openrouter_http_referer: str = "https://github.com/firefly-iii/firefly-iii-agent"
    openrouter_x_title: str = "firefly-bot"

    # --- Firefly III ---
    firefly_url: str = Field(..., pattern=r"^https?://")
    firefly_pat: str = Field(..., min_length=20)

    # --- Account defaults ---
    # Names rather than IDs so the same env file works across Firefly
    # instances (IDs differ between installs; names are user-controlled).
    # The bot resolves names → IDs at startup against the live Firefly
    # account list.
    default_asset_account_name: str = Field(..., min_length=1, max_length=200)
    liability_account_names: list[str] = Field(default_factory=list)

    default_currency: str = Field("IDR", pattern=r"^[A-Z]{3}$")
    secondary_currency: str = Field("USD", pattern=r"^[A-Z]{3}$")

    # IANA timezone name (e.g. "Asia/Jakarta", "Europe/Berlin", "UTC").
    # Used to stamp transactions with full ISO 8601 datetime including
    # offset, so Firefly III stops auto-translating from UTC midnight.
    timezone: str = Field("UTC", min_length=1, max_length=64)

    # --- Runtime behavior ---
    pending_ttl_minutes: int = Field(30, gt=0, le=1440)
    log_level: str = Field("INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # --- Paths ---
    state_db_path: str = "./data/state.db"
    config_toml_path: str = "./config.toml"

    @field_validator("telegram_owner_ids", mode="before")
    @classmethod
    def _parse_csv_ints(cls, v: object) -> list[int]:
        """Accept a comma-separated string from env, convert to list[int]."""
        if isinstance(v, str):
            if not v.strip():
                return []
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [int(x) for x in v]
        raise ValueError(f"expected str or list, got {type(v).__name__}")

    @field_validator("liability_account_names", mode="before")
    @classmethod
    def _parse_csv_strs(cls, v: object) -> list[str]:
        """Accept comma-separated string from env, convert to list[str]."""
        if isinstance(v, str):
            if not v.strip():
                return []
            return [x.strip() for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        raise ValueError(f"expected str or list, got {type(v).__name__}")

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        """Ensure the timezone string is a valid IANA name.

        Catches typos at startup (e.g. "Asia/Jakata" → fail fast)
        rather than letting the bot stamp every transaction with
        a broken offset.
        """
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as e:
            raise ValueError(
                f"TIMEZONE={v!r} is not a valid IANA timezone. "
                f"Examples: 'UTC', 'Asia/Jakarta', 'Europe/Berlin'."
            ) from e
        return v

    @classmethod
    def from_env(cls) -> EnvSettings:
        """Load settings from os.environ.

        Uppercases all field names to match env convention (pydantic field
        names are snake_case, env vars are SCREAMING_SNAKE_CASE).
        """
        raw: dict[str, object] = {}
        for field_name in cls.model_fields:
            env_name = field_name.upper()
            if env_name in os.environ:
                raw[field_name] = os.environ[env_name]
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            missing = [err["loc"][0] for err in e.errors() if err["type"] == "missing"]
            if missing:
                missing_envs = ", ".join(str(m).upper() for m in missing)
                raise SystemExit(
                    f"Missing required environment variables: {missing_envs}\n"
                    f"Check your .env file is loaded — see README §Setup. "
                    f"For Docker: `env_file: - .env` in docker-compose.yml. "
                    f"For systemd: EnvironmentFile= directive."
                ) from e
            raise SystemExit(f"Invalid configuration:\n{e}") from e


# ============================================================
# TOML-backed settings (behavioral config, ships with code)
# ============================================================


class CurrencyConfig(BaseModel):
    primary: str
    secondary: str
    allowed: list[str]


class LLMConfig(BaseModel):
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(800, gt=0, le=8000)
    timeout_seconds: int = Field(30, gt=0)
    models: list[str] = Field(..., min_length=1)


class FlowConfig(BaseModel):
    always_show_currency_toggle: bool = True
    top_accounts_in_keyboard: int = Field(3, ge=1, le=8)


class LoggingConfig(BaseModel):
    log_unauthorized_attempts: bool = True
    log_llm_raw_responses: bool = False
    log_firefly_request_bodies: bool = False


class TomlSettings(BaseModel):
    """Settings loaded from config.toml."""

    allowed_categories: list[str]
    tag_groups: dict[str, list[str]]
    currencies: CurrencyConfig
    # Maps merchant phrase → liability account NAME (resolved to ID at runtime)
    liability_keyword_names: dict[str, str]
    llm: LLMConfig
    flow: FlowConfig
    logging: LoggingConfig

    @classmethod
    def from_toml(cls, path: Path) -> TomlSettings:
        if not path.is_file():
            raise SystemExit(f"config.toml not found at {path}")
        with path.open("rb") as f:
            data = tomllib.load(f)

        try:
            return cls.model_validate(
                {
                    "allowed_categories": data["categories"]["allowed"],
                    "tag_groups": data["tags"],
                    "currencies": data["currencies"],
                    "liability_keyword_names": data.get("liabilities", {}).get("keywords", {}),
                    "llm": data["llm"],
                    "flow": data["flow"],
                    "logging": data["logging"],
                }
            )
        except KeyError as e:
            raise SystemExit(f"config.toml missing required section: {e}") from e
        except ValidationError as e:
            raise SystemExit(f"Invalid config.toml:\n{e}") from e


# ============================================================
# Combined settings — what the rest of the app imports
# ============================================================


class Settings(BaseModel):
    env: EnvSettings
    toml: TomlSettings

    @property
    def all_valid_tags(self) -> set[str]:
        """Flattened set of all tags across all groups."""
        return {tag for group in self.toml.tag_groups.values() for tag in group}

    def is_valid_tag(self, tag: str) -> bool:
        """Check if a tag is in the taxonomy. Accepts `trip:*` prefix."""
        if tag.startswith("trip:") and len(tag) > 5:
            return True
        return tag in self.all_valid_tags

    def is_valid_category(self, category: str) -> bool:
        return category in self.toml.allowed_categories

    def is_valid_currency(self, code: str) -> bool:
        return code in self.toml.currencies.allowed


def load_settings() -> Settings:
    """Top-level entry point — loads and validates everything.

    Call this once at startup. Propagates SystemExit with a clear message
    if anything's wrong.
    """
    env = EnvSettings.from_env()
    toml = TomlSettings.from_toml(Path(env.config_toml_path))
    log.info(
        "Settings loaded: %d owner(s), %d categories, %d tags, %d models",
        len(env.telegram_owner_ids),
        len(toml.allowed_categories),
        len({tag for group in toml.tag_groups.values() for tag in group}),
        len(toml.llm.models),
    )
    return Settings(env=env, toml=toml)
