"""BNPL (Buy Now Pay Later) detection.

Two related concerns:

1. **Liability lookup** — given a parsed transaction, does the merchant or
   the user's text match a BNPL provider keyword? If so, return the
   liability account ID to use as source/destination.

2. **Repayment intent cross-check** — even when BNPL is detected, is this
   actually a *repayment* of existing debt rather than a *new purchase*?
   The LLM produces an `intent` field; we cross-check it with keywords as
   defense-in-depth (LLM might mis-classify; keywords might miss; combined
   they're robust).

Returns None for liability lookup if not BNPL → normal asset-account flow.
Returns True/False for repayment intent.
"""

from __future__ import annotations

from firefly_agent.parsed import ParsedTransaction


# Words that strongly imply paying-down rather than buying-with.
# Multilingual: English + Bahasa Indonesia.
_REPAYMENT_KEYWORDS = (
    "repay",
    "repaid",
    "pay off",
    "paying off",
    "paid off",
    "settle",
    "settled",
    "settling",
    "bayar",      # Indonesian: pay
    "lunas",      # Indonesian: paid off
    "lunasi",     # Indonesian: pay off
    "dibayar",    # Indonesian: paid (passive)
    "cicilan",    # Indonesian: installment (often appears in repayment context)
)

# Words that imply buying-with. We don't strictly need a list of these —
# absence of repayment keywords is sufficient — but having them makes the
# cross-check more confident in disagreement cases.
_PURCHASE_KEYWORDS = (
    "bought",
    "buy",
    "buying",
    "purchased",
    "got",
    "ordered",
    "beli",       # Indonesian: buy
    "membeli",    # Indonesian: to buy
    "dibeli",     # Indonesian: bought (passive)
)


def detect_bnpl_account_id(
    parsed: ParsedTransaction,
    keyword_map: dict[str, int],
    full_text: str | None = None,
) -> int | None:
    """Match against BNPL keywords. Returns liability account ID if found.

    Checks (case-insensitive):
      1. `parsed.merchant`  — most reliable signal
      2. `full_text` if provided (the original user message)

    The keyword_map is loaded from config.toml `[liabilities.keywords]`.
    """
    if not keyword_map:
        return None

    haystacks: list[str] = []
    if parsed.merchant:
        haystacks.append(parsed.merchant.lower())
    if full_text:
        haystacks.append(full_text.lower())

    for keyword, account_id in keyword_map.items():
        kw_lower = keyword.lower()
        for hay in haystacks:
            if kw_lower in hay:
                return account_id

    return None


def detect_repayment_intent(
    parsed: ParsedTransaction,
    full_text: str | None = None,
) -> bool:
    """Return True if this transaction is a BNPL repayment, by keyword.

    This is a CROSS-CHECK against `parsed.intent`. The caller decides how
    to combine the two signals — typically: trust the LLM if the keyword
    check agrees or is silent; warn/fallback if they disagree.
    """
    haystacks: list[str] = []
    if parsed.description:
        haystacks.append(parsed.description.lower())
    if full_text:
        haystacks.append(full_text.lower())

    return any(kw in hay for kw in _REPAYMENT_KEYWORDS for hay in haystacks)


def reconcile_intent(
    parsed: ParsedTransaction,
    full_text: str | None = None,
) -> str:
    """Combine LLM-emitted intent with keyword cross-check.

    Returns the final intent: "purchase", "repayment", or "regular".

    Disagreement resolution:
      - If LLM says "repayment" AND keywords agree → "repayment"
      - If LLM says "repayment" AND keywords disagree → trust LLM (it has
        full context the keywords can't capture)
      - If LLM says "purchase"/"regular" AND keywords say repayment →
        prefer keywords (LLM may have missed a clear "bayar" / "repay")
      - Otherwise → trust LLM
    """
    llm_intent = parsed.intent
    kw_says_repayment = detect_repayment_intent(parsed, full_text=full_text)

    if llm_intent == "repayment":
        return "repayment"
    if kw_says_repayment and llm_intent in ("purchase", "regular"):
        # Override: the LLM might have classified "repay spaylater 1mil" as
        # "regular" if it didn't recognize the BNPL pattern. Keywords are
        # sharper here.
        return "repayment"
    return llm_intent