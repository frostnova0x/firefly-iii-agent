"""Service container — all long-lived singletons in one place.

Handlers access these via `context.bot_data[SERVICES_KEY]`.
The Application is constructed in bot.py, which builds a Services
instance and stores it in bot_data before starting polling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from firefly_agent.config import Settings
from firefly_agent.firefly import FireflyClient
from firefly_agent.openrouter import OpenRouterClient
from firefly_agent.security import OwnerGuard
from firefly_agent.state import StateStore

SERVICES_KEY = "services"


@dataclass
class Services:
    """Singleton bundle of clients and config.

    `liability_keyword_id_map` and `default_asset_account_id` are populated
    by `resolve_account_ids()` at bot startup — config holds names, runtime
    holds the IDs Firefly assigned to those names.
    """

    settings: Settings
    guard: OwnerGuard
    firefly: FireflyClient
    llm: OpenRouterClient
    store: StateStore
    # Resolved at startup:
    liability_keyword_id_map: dict[str, int] = field(default_factory=dict)
    default_asset_account_id: int = 0

    @staticmethod
    def get(context) -> Services:  # type: ignore[no-untyped-def]
        """Retrieve from a ContextTypes.DEFAULT_TYPE instance."""
        return context.bot_data[SERVICES_KEY]
