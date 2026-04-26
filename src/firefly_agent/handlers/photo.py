"""Photo message handler — receipt → LLM vision → preview keyboard.

Flow mirrors text.py:
    1. Whitelist (decorator)
    2. Download photo (largest size from Telegram)
    3. Resize to max 1024px wide via Pillow (saves tokens)
    4. Call llm.parse_transaction(image_bytes=...)
    5. Persist pending row
    6. Reply with preview + same keyboard

Resize limits:
    - Max 1024px on the long edge
    - JPEG quality 85
    - Strip EXIF metadata (rotation, GPS) — privacy + size

Token cost difference:
    Original 4000x3000 receipt: ~3000 image tokens
    Resized 1024x768:           ~600 image tokens
    Saving: ~80% on the image portion of the call.
"""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageOps
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from firefly_agent.errors import (
    OpenRouterAllModelsFailedError,
    OpenRouterAuthError,
    OpenRouterError,
)
from firefly_agent.formatting import format_error_message, format_preview_message
from firefly_agent.handlers.text import _build_initial_keyboard, _update_message_id
from firefly_agent.services import Services
from telegram.constants import ParseMode

log = logging.getLogger(__name__)

# Max long-edge in pixels; smaller = cheaper LLM call, faster upload
MAX_IMAGE_DIMENSION = 1024
JPEG_QUALITY = 85

# Telegram caption can include a hint like "yesterday" or "actually 50k not 45k"
MAX_CAPTION_LEN = 500


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Receipt photo handler. Whitelist already checked by decorator."""
    if update.message is None or update.effective_user is None or not update.message.photo:
        return

    services = Services.get(context)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id if update.effective_chat else user_id

    # Optional caption — user might add a hint like "lunch with sarah, reimbursable"
    caption = (update.message.caption or "").strip()[:MAX_CAPTION_LEN]

    # Telegram delivers multiple photo sizes; the LAST in the list is the largest.
    largest_photo = update.message.photo[-1]

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)

    # 1. Download
    try:
        tg_file = await context.bot.get_file(largest_photo.file_id)
        original_bytes = bytes(await tg_file.download_as_bytearray())
    except Exception as e:  # noqa: BLE001
        log.error("Telegram photo download failed: %s", e)
        await update.message.reply_text(
            format_error_message("Couldn't download the photo from Telegram."),
            parse_mode=ParseMode.HTML,
        )
        return

    # 2. Resize + re-encode
    try:
        resized_bytes, mime, w_orig, h_orig, w_new, h_new = _resize_image(original_bytes)
        log.info(
            "Receipt: %dx%d (%d bytes) → %dx%d (%d bytes)",
            w_orig, h_orig, len(original_bytes),
            w_new, h_new, len(resized_bytes),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Image resize failed (%s) — sending original", e)
        resized_bytes = original_bytes
        mime = "image/jpeg"

    # 3. Call the LLM
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        parsed = await services.llm.parse_transaction(
            text=caption if caption else None,
            image_bytes=resized_bytes,
            image_mime=mime,
        )
    except OpenRouterAuthError:
        log.error("OpenRouter auth failed — API key invalid")
        await update.message.reply_text(
            format_error_message(
                "LLM authentication failed. The operator needs to rotate the OpenRouter API key."
            ),
            parse_mode=ParseMode.HTML,
        )
        return
    except OpenRouterAllModelsFailedError as e:
        log.warning("All LLM models failed on photo: %d failures", len(e.failures))
        await update.message.reply_text(
            format_error_message(
                "All vision models are temporarily unavailable. Please try again in a minute."
            ),
            parse_mode=ParseMode.HTML,
        )
        return
    except OpenRouterError as e:
        log.warning("Vision parse failed: %s", e)
        await update.message.reply_text(
            format_error_message(
                "Couldn't read that receipt. If it's blurry or unusual, please type the transaction instead."
            ),
            parse_mode=ParseMode.HTML,
        )
        return

    # 4. Persist + reply (same code path as text.py from here on)
    payload_json = parsed.model_dump_json()
    cid = await services.store.insert_pending_transaction(
        user_id=user_id,
        chat_id=chat_id,
        message_id=0,
        payload_json=payload_json,
        currency=parsed.currency,
        ttl_minutes=services.settings.env.pending_ttl_minutes,
    )

    keyboard = await _build_initial_keyboard(services, cid, parsed)
    preview = format_preview_message(parsed)

    try:
        sent = await update.message.reply_text(
            preview,
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
    except Exception as e:  # noqa: BLE001
        log.error("Failed to send receipt preview: %s", e)
        await services.store.delete_pending_transaction(cid)
        raise

    await _update_message_id(services, cid, sent.message_id)

    log.info(
        "Receipt preview sent for user %d, callback_id=%s, "
        "merchant=%r, category=%s, confidence=%s",
        user_id, cid, parsed.merchant, parsed.category, parsed.confidence,
    )


def _resize_image(image_bytes: bytes) -> tuple[bytes, str, int, int, int, int]:
    """Open an image, normalize orientation, downscale, re-encode as JPEG.

    Returns: (encoded_bytes, mime_type, w_orig, h_orig, w_new, h_new)

    - Auto-rotates by EXIF before stripping EXIF (so the image renders correctly).
    - Strips all EXIF metadata (privacy: GPS, camera model, datetime).
    - Resizes proportionally so the long edge is <= MAX_IMAGE_DIMENSION.
    - Re-encodes as JPEG @ Q85.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Apply EXIF orientation BEFORE we throw the metadata away
        img = ImageOps.exif_transpose(img)
        # Convert any palette/RGBA/CMYK to RGB for clean JPEG output
        if img.mode != "RGB":
            img = img.convert("RGB")

        w_orig, h_orig = img.size

        # Compute new dims preserving aspect
        long_edge = max(w_orig, h_orig)
        if long_edge > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / long_edge
            w_new = int(round(w_orig * scale))
            h_new = int(round(h_orig * scale))
            img = img.resize((w_new, h_new), Image.Resampling.LANCZOS)
        else:
            w_new, h_new = w_orig, h_orig

        out = io.BytesIO()
        # No `exif=...` parameter → metadata is stripped by default
        img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    return out.getvalue(), "image/jpeg", w_orig, h_orig, w_new, h_new
