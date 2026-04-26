# firefly-iii-agent

A Telegram bot that turns natural-language messages and receipt photos into [Firefly III](https://www.firefly-iii.org/) transactions. Type `coffee 50k at starbucks` and the bot files it correctly. Send a photo of a receipt and it parses the merchant and total. No tapping through forms. Multilingual.

> **Status:** Production-ready for personal use. ~196 unit tests. Self-hosted, single-user by default.

---

## What it does

Send a message → get a preview with structured fields → tap Confirm → it lands in Firefly III.

```
You: coffee 50k at excelso
Bot: 📝 Preview
     Withdrawal: Rp 50,000
     Coffee at Excelso
     Food & Beverages · coffee
     2026-04-25
     [✓ IDR]  [USD]
     [🏦 BCA savings]
     [✏️ Edit]  [❌ Cancel]
You: [taps BCA savings]
Bot: Confirm? Withdrawal: Rp 50,000 from BCA savings ...
     [✅ Confirm]  [⬅️ Back]  [✏️ Edit]  [❌ Cancel]
You: [taps Confirm]
Bot: ✅ Logged. Firefly ID: 4
```

### Features

- **Natural language** — `coffee 50k at excelso`, `bayar listrik 350rb`, `$20 claude subscription`, `salary 15jt today`
- **Receipt photos** — send a photo, the LLM extracts merchant + total
- **Bilingual** — works in English and Bahasa Indonesia out of the box; tunable for other languages via the prompt
- **BNPL aware** — automatically distinguishes `bought headset 1mil spaylater` (a new debt) from `repay spaylater 1mil` (paying it down)
- **Currency-aware** — only shows accounts that match the transaction currency
- **Edit anything** — change description, merchant, tags, or notes from a sub-menu
- **Undo** — `/undo` then `/yes` to delete the last transaction
- **Whitelist-only** — only Telegram user IDs you list can use the bot. Strangers are silently ignored.

### What it doesn't do (yet)

- Recurring transaction templates
- Spending summaries / budgets in chat (read your Firefly UI for those)
- Multi-user collaboration

---

## Cost

The bot uses an LLM provider (OpenRouter) on a pay-per-request basis. Cost is calculated at 100tx/day with 50% of it being image scan:

| Model | Cost per month |
|---|---|
| `openai/gpt-5.4-nano` (default) | ~$1.64 |
| `openai/gpt-4o-mini` (cheaper alt) | ~$1.17 |
| Free-tier models | $0 (but expect occasional malformed responses) |

You'll need ~$5 of OpenRouter credit to start. Many free-tier models exist if you want $0/month, with the caveat that they sometimes fail to produce valid structured output.

---

## Requirements

- A Linux server (VPS, home server, or Raspberry Pi 4+)
- Docker + Docker Compose
- A running [Firefly III](https://docs.firefly-iii.org/) instance, OR willingness to spin one up (instructions below)
- A Telegram bot API
- An [OpenRouter](https://openrouter.ai) API
---

## Quick start

```bash
git clone https://github.com/frostnova0x/firefly-iii-agent.git
cd firefly-iii-agent
cp .env.example .env
cp config.toml.example config.toml
chmod 600 .env

# Edit .env with your values (see "Setup walkthrough" below)
nano .env

docker compose up -d
docker compose logs -f firefly-bot
```

If logs say `Account resolution OK` and `Started polling`, you're good. Open Telegram and send `/start` to your bot.

---

## Setup walkthrough

### 1. Get Firefly III running

If you already have Firefly III, skip this step.

If not, the simplest setup is the official Firefly III docker-compose. From [their docs](https://docs.firefly-iii.org/how-to/firefly-iii/installation/docker/):

```bash
mkdir -p ~/firefly-iii && cd ~/firefly-iii
curl -o docker-compose.yml https://raw.githubusercontent.com/firefly-iii/docker/main/docker-compose.yml
curl -o .env https://raw.githubusercontent.com/firefly-iii/docker/main/.env.example
nano .env  # Set APP_KEY (32 chars), DB_PASSWORD, and your timezone
docker compose up -d
```

Then open `http://your-server-ip` and create your admin account.

> **Recommendation:** install firefly-iii-agent on the same server as Firefly III. The bot only does outbound network calls, so it doesn't need any inbound port forwarding.

### 2. Set up your accounts in Firefly III

Before running the bot, create at least:

- **Asset accounts**: your bank/cash accounts (e.g. "BCA savings", "Cash wallet", "USD account"). Each has a fixed currency.
- **Liability accounts** (optional, only if you use BNPL or credit cards as debt): create a single liability account named `Account Payable` (or similar). Type = "Debt".
- **Categories**: create the categories you want the bot to be able to assign. Names must match what's in `config.toml` under `[categories.allowed]`.

### 3. Create a Telegram bot

1. Open [@BotFather](https://t.me/BotFather) in Telegram
2. Send `/newbot`
3. Choose a name (visible) and a username (must end in `bot`)
4. **Save the token** — looks like `1234567890:ABCdef...`. You'll need it in `.env`.
5. Optional but recommended: send `/setprivacy` → choose your bot → `Disable`. This lets the bot read all messages in private chats.

### 4. Get your Telegram user ID

You need your numeric Telegram user ID for the whitelist:

1. Open [@userinfobot](https://t.me/userinfobot) in Telegram
2. Send `/start`
3. The bot replies with your `Id:` — save this number

### 5. Get an OpenRouter API key

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Go to **Keys** → **Create Key**, save it
3. Go to **Credits** → add ~$5 (more than enough for several months)

### 6. Get a Firefly III Personal Access Token

1. In Firefly III, go to **Profile** → **OAuth**
2. Click **Personal Access Tokens** → **Create New Token**
3. Name it (e.g. "telegram-bot")
4. **Copy the long token immediately** — it's shown ONCE. Starts with `eyJ...`.

### 7. Configure `.env`

Open `.env` in your editor and fill in:

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdef...     # from BotFather
TELEGRAM_OWNER_IDS=123456789                # from @userinfobot
OPENROUTER_API_KEY=sk-or-v1-...             # from OpenRouter
FIREFLY_URL=http://firefly:8080             # or https://firefly.example.com
FIREFLY_PAT=eyJ0eXAiOiJKV1QiLCJ...          # from Firefly OAuth
DEFAULT_ASSET_ACCOUNT_NAME=BCA savings      # exact name from Firefly
LIABILITY_ACCOUNT_NAMES=Account Payable     # comma-separated, optional
DEFAULT_CURRENCY=IDR                        # 3-letter ISO
SECONDARY_CURRENCY=USD
```

> **Important:** make sure `.env` is `chmod 600` (owner-only). It contains every credential the bot has.

### 8. Configure `config.toml`

Open `config.toml` and edit:
- `[categories.allowed]` — match the category names in Firefly III
- `[currencies]` — set primary/secondary to your currencies
- `[liabilities.keywords]` — Indonesian BNPL providers are pre-loaded; replace with your own or remove if unused

### 9. Run the bot

```bash
docker compose up -d
docker compose logs -f firefly-bot
```

You should see:

```
Firefly III reachable: version 6.6.x
Account resolution OK: default_asset=BCA savings (id=6), BNPL keywords mapped: 7
Started polling.
```

If it fails, the error message tells you exactly which name doesn't match. Fix `.env` or `config.toml` and restart.

### 10. Test it

In Telegram, open your bot and send:

- `/start` — should greet you
- `/accounts` — lists your asset accounts
- `coffee 50k` — should parse and prompt for confirmation

---

## VPS port-forwarding (optional)

If you want to **access Firefly III from outside your VPS**, you'll need to expose its port. **The bot itself does NOT need any port forwarding** — Telegram uses outbound long-polling, so the bot reaches out to Telegram, not the other way around.

For Firefly III's web UI, the simplest path is `ufw`:

```bash
# Allow SSH (probably already done)
sudo ufw allow 22/tcp

# Allow HTTPS (if Firefly is on port 443 via reverse proxy)
sudo ufw allow 443/tcp

# Allow HTTP (only if you don't have HTTPS yet — switch to HTTPS ASAP)
sudo ufw allow 80/tcp

sudo ufw enable
sudo ufw status
```

Or with `iptables`:

```bash
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A INPUT -j DROP
sudo netfilter-persistent save  # if you have iptables-persistent
```

> **Strong recommendation:** put Firefly III behind [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) instead of opening ports. It's free, gives you HTTPS automatically, and means **no inbound ports open at all** — your VPS becomes invisible to internet scanners. The bot still works fine because it only needs outbound traffic to `api.telegram.org`, `openrouter.ai`, and your Firefly URL.

If Firefly is on the same Docker network as the bot (same `docker-compose.yml`), the bot reaches Firefly on the internal network (`http://firefly:8080`) — you don't need to expose Firefly's port at all unless you want external access.

---

## Updating

```bash
cd firefly-iii-agent
git pull
docker compose build --no-cache firefly-bot
docker compose up -d firefly-bot
```

Database migrations run automatically on startup. Your `data/state.db` is preserved across rebuilds.

---

## Troubleshooting

### "Bot doesn't respond to my messages"

- Check `docker compose logs -f firefly-bot`. Look for "Started polling" — that means it's running. If not, scroll up for the actual error.
- Confirm your `TELEGRAM_OWNER_IDS` matches your real ID (not someone else's). Anyone not in this list is silently ignored — no error, no reply.
- Confirm the bot's privacy is **disabled** (BotFather → `/setprivacy`).

### "Account resolution failed"

The startup log will tell you exactly which name doesn't match. Two common causes:
- Typo in `.env` (`DEFAULT_ASSET_ACCOUNT_NAME=BCA savign` — misspelled)
- Typo in `config.toml` (`[liabilities.keywords]` references a name that doesn't exist)
- Account exists but is *deactivated* in Firefly — the API doesn't return deactivated accounts

### "Firefly preflight failed"

- Check `FIREFLY_URL` is reachable from inside the Docker container. From the host: `curl $FIREFLY_URL/api/v1/about` should work (with the PAT as a header). If not, the URL is wrong or there's a network problem.
- If Firefly is on the same Docker network as the bot, use the Docker service name (`http://firefly:8080`), not `localhost`.
- If you put Firefly behind a reverse proxy with HTTPS, make sure the proxy passes through API requests at `/api/v1/...`.

### "Firefly rejected the transaction"

- Look at the field name in the error. Most common: a category that doesn't exist in Firefly. The bot's prompt is restricted to your `[categories.allowed]` list, but Firefly is the source of truth.
- For BNPL repayments specifically: Firefly models asset → liability movements as **withdrawals**, not transfers. The bot already does this correctly; if it fails, your liability account might be the wrong type.

### "I want to see what the bot is doing"

```bash
LOG_LEVEL=DEBUG docker compose up -d
docker compose logs -f firefly-bot
```

DEBUG logs each LLM call, each Firefly API call, and intent reconciliation decisions.

---

## Privacy & security

### What the bot sees

- Every text message and photo you send to it
- Your Firefly III account names, transaction descriptions, categories

### What goes to OpenRouter (the LLM)

- The text of each transaction message OR the bytes of each receipt photo
- The list of your asset/liability account names (passed in the system prompt for context)
- The list of your categories and tag groups
- **Your raw token, PAT, or Firefly URL is NEVER sent to OpenRouter.** Those stay local to the bot.

OpenRouter forwards each request to whichever underlying model you chose. Read [OpenRouter's privacy policy](https://openrouter.ai/privacy) and the underlying provider's policy. By default, OpenAI does NOT train on API requests.

### What goes to Telegram

- Every message you send the bot is on Telegram's servers (it's a Telegram bot — there's no way around this)
- Telegram does not train on bot conversations as of this writing, but you should not assume this is permanent

### What stays local to the bot

- Your `.env` file (Telegram token, API keys, Firefly PAT)
- The SQLite state database (pending transactions, account usage stats)
- Logs

### Hardening recommendations

- Run on a non-root user (the Dockerfile already does this — but double-check if you go bare-metal)
- `chmod 600 .env` always — anyone who can read this owns your finances
- Keep `LIABILITY_ACCOUNT_NAMES` minimal — only list accounts the bot is authorized to touch
- Rotate your Firefly PAT periodically (Firefly's UI lets you delete and recreate)
- Don't reuse `.env` across servers; one server, one `.env`

---

## Contributing

Bug reports and pull requests welcome. Before sending a PR:

```bash
uv sync --dev
uv run pytest tests/ --ignore=tests/test_m1_2_integration.py --ignore=tests/test_m1_3_integration.py
```

Tests should pass. Integration tests require a real Firefly III + a `FIREFLY_INTEGRATION_OK=1` env var.

---

## License

[AGPL-3.0](LICENSE) — you're free to use, modify, and self-host. If you run a modified version as a network service (SaaS), you must publish your modifications under the same license.

---

## Acknowledgements

- [Firefly III](https://www.firefly-iii.org/) — the genuinely good self-hosted finance manager that makes this bot possible
- [python-telegram-bot](https://python-telegram-bot.org/)
- [OpenRouter](https://openrouter.ai)
- [astral-sh/uv](https://github.com/astral-sh/uv)
