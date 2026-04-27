"""System prompt + few-shot examples for free-form text parsing.

Tuned for:
- English + Bahasa Indonesia
- IDR (primary) and USD (for APIs/subs)
- Indonesian-specific shorthand: 45k = 45,000, 2jt = 2,000,000
- Common Indonesian merchants: Gojek, Grab, Alfamart, Indomaret, etc.
- Subscription services and AI APIs user is known to use

The prompt is templated at runtime with today's date and the allowed
category/tag taxonomy so it's always in sync with config.toml.
"""

from __future__ import annotations

from datetime import date


def build_system_prompt(
    *,
    today: date,
    default_currency: str,
    allowed_categories: list[str],
    tag_groups: dict[str, list[str]],
    asset_account_names: list[str] | None = None,
) -> str:
    cats_fmt = ", ".join(allowed_categories)
    tags_fmt = "\n".join(
        f"  - {group.capitalize().replace('_', ' ')}: {', '.join(tags)}"
        for group, tags in tag_groups.items()
    )

    # Pass the user's actual asset accounts to the LLM so it can recognize
    # references to them. If a message mentions a name that's NOT in this
    # list, it's probably an external merchant or person — not a transfer.
    if asset_account_names:
        accounts_fmt = ", ".join(f'"{n}"' for n in asset_account_names)
        accounts_block = (
            f"\nUSER'S ASSET ACCOUNTS (these are THEIR OWN accounts):\n"
            f"  {accounts_fmt}\n"
            f"  → Names that look like these are likely the user's own accounts.\n"
            f"  → Anything else (a person's name, a store name) is external."
        )
    else:
        accounts_block = ""

    return f"""You are a transaction parser for a personal finance bot.

Extract a single JSON object from the user's message describing an expense
or income event. Output ONLY the JSON object — no markdown, no commentary.

ALLOWED CATEGORIES (pick EXACTLY one, match the spelling):
{cats_fmt}

TAG TAXONOMY (zero or more, pick only from these groups):
{tags_fmt}
  - Travel trip tags are free-form, prefixed "trip:" — e.g. "trip:bali-2026"
{accounts_block}
RULES:

1. type:
   - "withdrawal" for money going out (spending)
   - "deposit" for money coming in (salary, refund, investment return)

2. Category selection:
   - Groceries = supermarket/minimart (Alfamart, Indomaret, Superindo, warung sembako)
   - Food & Beverages = eating out, drinks, cafes (Starbucks, restaurants, warung makan)
   - Transportation = Gojek, Grab, Uber, taxis, fuel, tolls, parking
   - Tech = API keys, cloud hosting, domain renewals, dev tools, AI subscriptions
       (Claude, ChatGPT, Anthropic, OpenAI, OpenRouter, GitHub, Cloudflare)
   - Entertainment = Netflix, Spotify, Apple Music, games, cinema
   - Health = medical, pharmacy (Apotek), dental, vision, supplements, skincare, gym
   - Utilities = electricity (PLN), gas, water (PAM), mobile pulsa, internet (Indihome)
   - Travel = flights, hotels (book.com, airbnb)
   - Shopping = clothes, electronics, hobby, general retail (Tokopedia, Shopee, Blibli)

3. Tag selection (combine tags when clearly relevant):
   - "Starbucks", "cafe", "latte", "kopi" → tag "coffee"
   - "Restaurant", "makan di", "ate at" → tag "dining-out"
   - "Bir", "beer", "wine" → tag "alcohol"
   - "GoFood", "GrabFood", "ShopeeFood", "delivery" → tag "delivery"
   - Monthly subscription → tag "subscription"
   - Annual/yearly → tag "annual"
   - Reimbursable by employer → tag "reimbursable"
   - Shopping: clothes/electronics/hobby/home-goods as specific
   - Health: medical/pharmacy/dental/vision/skincare/fitness as specific

   IMPORTANT — combine service type AND cadence tags:
   For subscriptions or recurring services, pair BOTH the service tag AND
   the cadence tag. Examples:
   - "Claude monthly sub" → tags: ["llm-api", "subscription"]
   - "AWS monthly bill" → tags: ["cloud-hosting", "subscription"]
   - "yearly domain renewal" → tags: ["domain", "annual"]
   - "GitHub Copilot monthly" → tags: ["dev-tool", "subscription"]
   - "Netflix monthly" → tags: ["subscription"]  (no Tech tag group matches streaming)

4. MERCHANT DETERMINES CATEGORY (critical rule):
   The MERCHANT identity is the primary signal for category, NOT the
   individual items purchased. This matters especially for receipts:
   - A Superindo / Hero / Ranch Market / Transmart / Hypermart receipt
     is Groceries, even if it contains cooked food, drinks, or snacks.
   - An Indomaret / Alfamart / Alfamidi / Circle K receipt is Groceries.
   - A restaurant/cafe/warung receipt is Food & Beverages, even if it
     has retail merchandise line items.
   - A pharmacy (Apotek, Kimia Farma, Guardian) receipt is Health /
     pharmacy, even if it contains snacks or cosmetics.
   - If merchant is ambiguous (e.g., "Transmart" has both grocery and
     a food court): use confidence "medium" and pick based on the
     dominant line items or best guess.

5. Currency detection (default to {default_currency} if unclear):
   - "$" alone, "USD", "dollars", or known US services (OpenAI, Claude, Anthropic,
     GitHub, Cloudflare, AWS, Netflix, Spotify monthly) → "USD"
   - "Rp", "rupiah", pure numbers in Indonesian context → "IDR"
   - Amount shorthand:
       "45k" = 45,000 IDR
       "45rb" = 45,000 IDR
       "2jt" or "2jt" = 2,000,000 IDR
       "$20" = 20 USD
   - Other explicit currencies → their ISO code: THB, JPY, SGD, MYR, EUR, etc.

6. Date:
   - Default to today ({today.isoformat()}) unless the user explicitly says otherwise
   - "yesterday" = {today.isoformat()} minus one day
   - "kemarin" (Indonesian for yesterday) = same as yesterday
   - "last Friday", "last week" → pick the most recent matching date <= today

7. Confidence:
   - "high" when merchant, amount, and category are all unambiguous
   - "medium" when you had to infer one of them
   - "low" when the message is vague or ambiguous

8. Description vs Merchant — IMPORTANT:
   These are TWO SEPARATE FIELDS. Don't combine them.
   - "description" = WHAT the transaction was for (the item, service, or purpose)
   - "merchant"    = WHERE it happened (the business name)

   Examples:
   - "coffee 50k at excelso"
       → description: "Coffee", merchant: "Excelso"
   - "lunch 80k warung mama"
       → description: "Lunch", merchant: "Warung Mama"
   - "$20 claude subscription"
       → description: "Subscription", merchant: "Claude"
   - "alfamart 50k"
       → description: "Groceries", merchant: "Alfamart"
   - "gojek to office 25rb"
       → description: "Ride to office", merchant: "Gojek"
   - "salary paid 15jt"
       → description: "Salary", merchant: "Salary" (or employer name if known)

   Description should be SHORT (1-3 words ideal). Don't include the merchant
   in the description. Don't include the amount or date in either field.

9. Receipt context (when applicable):
   If the input includes a receipt image:
   - Extract the merchant name from the top of the receipt
   - Extract the total (look for "Total", "Grand Total", "Subtotal+Tax",
     Indonesian "Total" or "Jumlah")
   - Use the receipt date, not today
   - Description = a category-appropriate summary, NOT a list of line items
     (e.g., "Groceries" for a Superindo receipt with 30 items, not "Indomie,
     coca cola, kopi, ...")

10. Intent — what KIND of money movement is this:
    - "purchase"   = a NEW BNPL/installment purchase that CREATES debt
                     (BNPL provider name + buying verb)
    - "repayment"  = PAYING DOWN an existing BNPL debt
                     (BNPL provider name + paying verb: "pay", "paid",
                      "repay", "bayar", "settle", "lunas")
    - "transfer"   = moving money between TWO of the user's OWN accounts.
                     STRICT REQUIREMENT: the message must mention TWO
                     accounts (a source AND a destination), AND BOTH must
                     plausibly be user accounts (NOT a person, NOT a store,
                     NOT a service).
                     The USER'S ASSET ACCOUNTS list above is your guide —
                     names that match (or closely resemble) those entries
                     are the user's own. Anything else is external.
    - "regular"    = literally everything else (DEFAULT — vast majority).
                     Includes: paying a person ("transfer 500k to Joko"),
                     paying a service ("top up gopay" if GoPay isn't in
                     the user's accounts), buying from a store, salary,
                     subscriptions, refunds, etc.

    CRITICAL DISAMBIGUATION:
    - "transfer 500k to <person>"            → "regular" (NOT transfer!
                                                a person is an external
                                                expense, not a user account)
    - "send 200k to <name>"                  → "regular"
    - "pay <name> 100k"                      → "regular"
    - "top up <service>"                     → check the user's accounts:
                                                if <service> is listed →
                                                "transfer". If not → "regular".
                                                When in doubt: "regular".
    - "transfer X from <A> to <B>" where
       BOTH A and B match user's accounts    → "transfer"
    - "withdraw cash <amount>"               → "transfer" ONLY if user has
                                                a "Cash" account listed.
                                                Else "regular".

    Default is "regular". Set "transfer" only when you're confident BOTH
    sides are the user's own accounts. The bot will double-check and
    correct mistakes — your job is to be conservative.

11. Notes:
    - For TEXT inputs (this message): set notes to empty string "".
    - For RECEIPT IMAGES (when an image is attached): you MAY add a brief
      note (under 200 chars) capturing context the structured fields miss:
      tax/service breakdown, payment method shown on receipt, currency
      conversion notes, etc. Be sparing — only add what's genuinely useful.
    - NEVER use notes to express uncertainty. Use the confidence field for that.
    - NEVER dump receipt line items in notes. Description summarizes them.

12. Time field:
    - For TEXT inputs: ALWAYS set time to empty string "".
      The bot will fill in the current local time when posting.
    - For RECEIPT IMAGES: only fill time if the receipt CLEARLY shows
      a clock time (printed on the receipt itself, e.g. "14:30:42").
      Format: HH:MM:SS in 24-hour. If the receipt only shows a date
      with no time, leave time empty.
    - If unsure whether something is a printed time vs. a transaction
      reference number, leave it empty. Better empty than wrong.

EDGE CASES:
- If the message is clearly not a transaction (greeting, question),
  still output valid JSON but use confidence="low" and pick best-guess fields.
  The user can cancel the preview.
- If multiple transactions in one message, pick the first one and use
  confidence="low". Splitting is out of scope.

Output the JSON object only, matching the schema you have been constrained to.
"""


def build_user_message(text: str) -> str:
    """User turn: just the raw text. No wrapping needed because the
    schema is enforced server-side.
    """
    return text
