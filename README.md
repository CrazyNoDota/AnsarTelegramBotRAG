# Ansar Telegram Study Bot — Grants RAG

Telegram bot that answers questions about a database of grant programs
(`data/grants.xlsx`) using retrieval-augmented generation. Designed to
**never hallucinate** — every answer is grounded in retrieved rows, and
the bot openly says "I don't have this information" when the data isn't
in the index.

## Architecture

```
User question
    │
    ▼
Telegram → Vercel webhook (api/webhook.py)
    │
    ▼
bot.telegram_handler.handle_update
    │
    ├── /start, /help, /count, /reload
    │
    └── plain text → bot.rag.retrieve  (top-5 cosine over OpenAI embeddings)
                          │
                          ▼
                   bot.llm.answer  (NVIDIA Qwen → OpenAI fallback)
                          │
                          ▼
                  reply with citations
```

- **Embeddings:** OpenAI `text-embedding-3-small` (1536d, multilingual)
- **Vector store:** L2-normalized numpy matrix saved as `.npz`. Swap to
  FAISS only if the corpus exceeds ~50k rows.
- **LLM:** NVIDIA-hosted Qwen (free tier) primary, OpenAI `gpt-4o-mini` fallback.
- **Anti-hallucination:** strict system prompt, retrieval-only context,
  fixed refusal phrase if no relevant grant found.

## Repo layout

```
api/webhook.py           Vercel serverless entrypoint
bot/data.py              xlsx → Grant dataclass
bot/embed.py             OpenAI embeddings wrapper
bot/rag.py               retrieval (cosine top-k)
bot/llm.py               Qwen + OpenAI fallback
bot/telegram_handler.py  command dispatcher
scripts/build_index.py   rebuild data/index.npz + chunks.json
data/grants.xlsx         source spreadsheet
data/chunks.json         pre-built chunks (committed)
data/index.npz           pre-built embeddings (committed)
```

## Local development

```bash
git clone https://gitlab.com/akdzhalil/ansartelegramstudybot.git
cd ansartelegramstudybot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, NVIDIA_API_KEY, ADMIN_CHAT_IDS

# build the embeddings index (only needed when grants.xlsx changes)
python scripts/build_index.py

# quick sanity check
python -m bot.data data/grants.xlsx
```

## Updating the grants

1. Edit `data/grants.xlsx` — add rows, never reorder columns.
2. Run `python scripts/build_index.py` to rebuild the index.
3. `git add data/ && git commit -m "update grants" && git push`.
4. Vercel auto-redeploys. Or send `/reload` in Telegram for hot-reload
   (only works if you committed; the file system on Vercel is read-only
   between deploys, so /reload is mostly useful for self-hosted runs).

## Deploy to Vercel

1. Import the GitLab repo into Vercel: https://vercel.com/new
2. Set environment variables in the Vercel project settings:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `NVIDIA_API_KEY` (optional)
   - `ADMIN_CHAT_IDS`
3. Deploy. You'll get a URL like `https://ansartelegramstudybot.vercel.app`.
4. Register the webhook with Telegram **once**:

   ```bash
   curl -F "url=https://<your-vercel-domain>/api/webhook" \
        https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook
   ```

5. Verify with:

   ```bash
   curl https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getWebhookInfo
   ```

## Commands

| Command   | Who      | What                                        |
| --------- | -------- | ------------------------------------------- |
| `/start`  | anyone   | Welcome message and grant count             |
| `/help`   | anyone   | List commands                               |
| `/count`  | anyone   | Number of grants currently indexed          |
| `/reload` | admin    | Reread `data/index.npz` without redeploy    |

Any other text is treated as a question and answered from the grants DB.

## Security notes

- The Telegram bot token and the NVIDIA key originally shared during
  setup must be **rotated**. Treat any secret that has appeared in chat,
  git history, or screenshots as compromised.
- Never commit `.env`. The `.gitignore` already excludes it.
- `ADMIN_CHAT_IDS` should contain only Telegram user IDs you trust;
  `/reload` is safe but it's the entry point for any future admin tools.

## License

Private project — contact the owner before reusing.
