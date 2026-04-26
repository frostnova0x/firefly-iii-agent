"""Entry point: `python -m firefly_agent` or the `firefly-agent` console script.

Loads config, configures logging, runs the bot via asyncio.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from firefly_agent.bot import run_bot
from firefly_agent.config import load_settings


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet down chatty libraries unless explicit DEBUG
    if level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("telegram.ext").setLevel(logging.INFO)


def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    _configure_logging(log_level)

    log = logging.getLogger("firefly_agent")
    log.info("firefly-iii-agent starting up…")

    settings = load_settings()
    log.info(
        "Configured: firefly=%s, models=%s, owners=%s, default_currency=%s",
        settings.env.firefly_url,
        settings.toml.llm.models,
        settings.env.telegram_owner_ids,
        settings.env.default_currency,
    )

    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except SystemExit:
        raise
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
