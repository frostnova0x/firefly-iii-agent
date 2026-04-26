"""Bot assembly — construct everything and run polling.

Called from __main__.py. Long-lived: runs until shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from firefly_agent.config import Settings
from firefly_agent.firefly import FireflyClient
from firefly_agent.handlers.callbacks import handle_callback
from firefly_agent.handlers.commands import (
    cmd_accounts,
    cmd_balance,
    cmd_cancel,
    cmd_start,
    cmd_undo,
    cmd_yes,
)
from firefly_agent.handlers.photo import handle_photo_message
from firefly_agent.handlers.text import handle_text_message
from firefly_agent.openrouter import OpenRouterClient
from firefly_agent.security import OwnerGuard, require_owner
from firefly_agent.services import SERVICES_KEY, Services
from firefly_agent.state import StateStore

log = logging.getLogger(__name__)

REAP_INTERVAL_SECONDS = 300


async def _resolve_account_names(services: Services) -> None:
    """Resolve config-by-NAME → live Firefly account IDs.

    Mutates the services object. Raises SystemExit (= bot won't start)
    if any configured name doesn't match an account in Firefly. This is
    intentional: a typo in the config would otherwise cause BNPL or
    default-account routing to silently break for weeks before being
    noticed.
    """
    env = services.settings.env
    asset_accounts = await services.firefly.list_asset_accounts()
    liability_accounts = await services.firefly.list_liability_accounts()

    # Build name → id maps, normalizing case for forgiveness on user typos
    asset_by_name = {a.name.casefold(): a for a in asset_accounts}
    liability_by_name = {a.name.casefold(): a for a in liability_accounts}

    # 1. Default asset account
    default_key = env.default_asset_account_name.casefold()
    default_acc = asset_by_name.get(default_key)
    if default_acc is None:
        valid_names = sorted(a.name for a in asset_accounts)
        raise SystemExit(
            f"DEFAULT_ASSET_ACCOUNT_NAME='{env.default_asset_account_name}' "
            f"not found in Firefly III. "
            f"Available asset accounts: {valid_names}"
        )
    services.default_asset_account_id = default_acc.id

    # 2. BNPL keyword → liability_account name → liability_account id
    keyword_id_map: dict[str, int] = {}
    missing: list[tuple[str, str]] = []  # [(keyword, name), ...]
    for keyword, name in services.settings.toml.liability_keyword_names.items():
        liability = liability_by_name.get(name.casefold())
        if liability is None:
            missing.append((keyword, name))
            continue
        keyword_id_map[keyword] = liability.id

    if missing:
        valid_names = sorted(a.name for a in liability_accounts)
        details = ", ".join(f"'{kw}' → '{name}'" for kw, name in missing)
        raise SystemExit(
            f"BNPL keyword(s) reference unknown liability accounts: {details}. "
            f"Available liability accounts: {valid_names}. "
            f"Fix [liabilities.keywords] in config.toml or create the missing "
            f"accounts in Firefly III."
        )
    services.liability_keyword_id_map = keyword_id_map

    log.info(
        "Account resolution OK: default_asset=%s (id=%d), "
        "BNPL keywords mapped: %d",
        default_acc.name, default_acc.id, len(keyword_id_map),
    )


async def _reap_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic cleanup of expired pending transactions and edit-mode rows."""
    services: Services = context.bot_data[SERVICES_KEY]
    try:
        await services.store.reap_expired()
        await services.store.reap_expired_edit_modes()
    except Exception as e:  # noqa: BLE001
        log.warning("Reap job failed: %s", e)


async def run_bot(settings: Settings) -> None:
    """Main entry — construct services, wire handlers, start polling."""
    guard = OwnerGuard(
        owner_ids=settings.env.telegram_owner_ids,
        log_unauthorized=settings.toml.logging.log_unauthorized_attempts,
    )

    async with AsyncExitStack() as stack:
        firefly = await stack.enter_async_context(
            FireflyClient(
                base_url=settings.env.firefly_url,
                personal_access_token=settings.env.firefly_pat,
            )
        )
        llm = await stack.enter_async_context(
            OpenRouterClient(
                api_key=settings.env.openrouter_api_key,
                models=settings.toml.llm.models,
                allowed_categories=settings.toml.allowed_categories,
                allowed_tags=sorted(settings.all_valid_tags),
                allowed_currencies=settings.toml.currencies.allowed,
                default_currency=settings.env.default_currency,
                tag_groups=settings.toml.tag_groups,
                http_referer=settings.env.openrouter_http_referer,
                x_title=settings.env.openrouter_x_title,
                temperature=settings.toml.llm.temperature,
                max_tokens=settings.toml.llm.max_tokens,
                timeout_seconds=settings.toml.llm.timeout_seconds,
            )
        )
        store = await stack.enter_async_context(StateStore(settings.env.state_db_path))

        services = Services(
            settings=settings,
            guard=guard,
            firefly=firefly,
            llm=llm,
            store=store,
        )

        # Startup preflight: verify Firefly is reachable with this PAT.
        try:
            about = await firefly.about()
            log.info("Firefly III reachable: version %s", about.get("version", "?"))
        except Exception as e:  # noqa: BLE001
            log.error("Firefly preflight failed: %s", e)
            raise

        # Resolve config-by-NAME values to runtime IDs against the live
        # Firefly account list. Fail loudly if any configured name is missing
        # — silent fallback would mean BNPL transactions get mis-categorized
        # for weeks before anyone notices.
        try:
            await _resolve_account_names(services)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            log.error("Account name resolution failed: %s", e)
            raise SystemExit(
                f"Failed to resolve configured account names: {e}"
            ) from e

        app = Application.builder().token(settings.env.telegram_bot_token).build()
        app.bot_data[SERVICES_KEY] = services

        # Every handler is whitelisted. Unauthorized users are silently dropped.
        app.add_handler(CommandHandler("start", require_owner(guard)(cmd_start)))
        app.add_handler(CommandHandler("accounts", require_owner(guard)(cmd_accounts)))
        app.add_handler(CommandHandler("balance", require_owner(guard)(cmd_balance)))
        app.add_handler(CommandHandler("cancel", require_owner(guard)(cmd_cancel)))
        app.add_handler(CommandHandler("undo", require_owner(guard)(cmd_undo)))
        app.add_handler(CommandHandler("yes", require_owner(guard)(cmd_yes)))
        app.add_handler(CallbackQueryHandler(require_owner(guard)(handle_callback)))
        # Photo handler — receipts. Goes before text handler.
        app.add_handler(
            MessageHandler(
                filters.PHOTO,
                require_owner(guard)(handle_photo_message),
            )
        )
        # Text handler goes LAST so commands and photos match first.
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                require_owner(guard)(handle_text_message),
            )
        )

        if app.job_queue is not None:
            app.job_queue.run_repeating(
                _reap_job,
                interval=REAP_INTERVAL_SECONDS,
                first=REAP_INTERVAL_SECONDS,
                name="reap_expired",
            )
        else:
            log.warning(
                "JobQueue not available — expired pendings will only be cleaned "
                "opportunistically on insert."
            )

        log.info("Bot ready — starting long polling.")
        try:
            await app.initialize()
            await app.start()
            if app.updater is not None:
                await app.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True,
                )
            await asyncio.Event().wait()  # block until cancelled
        finally:
            log.info("Shutting down…")
            if app.updater is not None and app.updater.running:
                await app.updater.stop()
            await app.stop()
            await app.shutdown()
