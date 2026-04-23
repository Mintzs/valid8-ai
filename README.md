<div align="center">

# 🔍 Valid8

**A market research and product validation AI agent that tells you if your idea is worth building — before you build it.**

> Founders spend months building things nobody wants. Valid8 gives you **sentiment analysis** signals from real people on Reddit, Hacker News, and cross-platform sources in seconds. Not marketing fluff. Not fabricated stats. Actual demand evidence.

https://github.com/user-attachments/assets/ded53f11-a03e-4418-a277-f9f92c320f2a

</div>

***

## What is Valid8?

Valid8 is an **agentic market research assistant** powered by a large language model with real tool use. You describe your product idea and the agent:

1. Scrapes Reddit discussions across relevant subreddits
2. Searches Hacker News for matching threads and Ask HN posts
3. Researches competitor apps on Product Hunt, Google Play, and the App Store
4. Supplements findings with live DuckDuckGo web search
5. Synthesizes everything into a **founder-style stakeholder briefing**

No spreadsheets. No manual searching. No hallucinated market size stats.

The agent uses **VADER sentiment analysis** with custom demand-signal heuristics (buy-intent phrases, pain-point phrases, engagement weighting) to score your idea objectively, then interprets those scores into actionable guidance.

***

## Why This Exists

The typical validation workflow is:

- Google your idea manually
- Browse Reddit for 30 minutes
- Read a few Product Hunt comments
- Guess whether people would pay

Valid8 replaces that with a single prompt. The agent doesn't just return scores, it synthesizes what people are *actually* saying, where the strongest signal came from, whether the pain sounds annoying or mission-critical, and what you should do next.

Standard LLMs will answer "is this a good idea?" with generic encouragement or hallucinated data. Valid8 refuses to do that. If the data is weak, it says so. If platforms disagree, it surfaces the tension. Every claim is grounded in real posts, comments, and reviews collected at runtime.

### Verdict vs. Raw Score

| Raw tool output | Valid8 output |
|---|---|
| `Demand Score: 71 / Reddit sentiment: 48% positive` | *"Reddit shows repeated operational frustration among solo founders. HN is quieter — the problem is real, but not yet urgently framed."* |
| `3 competitors found on Play Store` | *"Competitors exist but reviews surface consistent onboarding complaints — a clear feature gap worth targeting."* |

***

## Tools & Data Sources

| Tool | Source | What It Captures |
|---|---|---|
| `analyze_reddit` | Reddit (public API) | Posts + comments across targeted subreddits, engagement-weighted sentiment |
| `hackernews_market_research` | HN Algolia API | Stories, Ask HN threads, comments — high signal for SaaS and dev tools |
| `competitor_research` | Product Hunt + Google Play + App Store | Competitor traction, user reviews, feature gaps, complaint patterns |
| `web_search` | DuckDuckGo Search | Category maturity, pricing patterns, trend confirmation, non-app competitors |

***

## How the Scoring Works

Each tool collects posts, comments, and reviews, then scores them using:

- **VADER** compound sentiment score (-1.0 to +1.0)
- **Buy-intent signal bonus** — phrases like `"take my money"`, `"where can I buy"`, `"need this"` each add +0.15
- **Pain-point signal bonus** — phrases like `"frustrated"`, `"can't find"`, `"broken"` each add +0.08
- **Engagement weight** — upvotes and comment counts amplify high-signal posts (capped to prevent single-post dominance)

Final demand score:

```
demand_score = clamp((avg_weighted_sentiment + 1) × 50, 0, 100)
```

| Score | Verdict |
|---|---|
| ≥ 70 | **Promising** |
| 45–69 | **Needs validation** |
| < 45 | **Weak demand** |

***

## Response Structure

Every full validation request produces a stakeholder briefing:

1. **Verdict** — direct 1–3 sentence conclusion upfront, no burying the lead
2. **Why** — interpretation of what the evidence actually means
3. **Social signals** — Reddit, HN, and competitor findings synthesized
4. **Market implications** — saturation level, differentiation angle, positioning
5. **Risks / Caveats** — where the data is weak or contradictory
6. **Recommended next move** — concrete action for the founder

The agent stays strictly in scope. It refuses coding help, general trivia, and anything outside market research — and politely redirects back to what it can do.

***

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Mintzs/valid8-ai.git
cd valid8-ai
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
PRODUCT_HUNT_API_KEY=your_product_hunt_key_here   # optional — improves competitor data
```

> Only `OPENROUTER_API_KEY` is required. Product Hunt credentials are optional but recommended for competitor research.

### 3. Run the agent (CLI)

```bash
python valid8.py
```

The agent runs as a terminal chat loop with persistent conversation memory:

```
You: I want to build a tool that helps solo founders track user feedback from multiple channels.

Assistant: There is real demand here, but the pain is fragmented rather than acute...
          [full stakeholder briefing follows]
```

### 4. Run with web UI (locally)

Start the FastAPI backend:

```bash
uvicorn app:app --reload
```

Then in a separate terminal, start the frontend dev server (requires Node.js):

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at http://localhost:5173 and connects to the FastAPI backend via a streaming SSE /chat endpoint.

***

## Project Structure

```
valid8-ai/
├── valid8.py          # Agent core — tools, scoring, system prompt, CLI loop
├── app.py             # FastAPI web server with streaming
├── frontend/          # Web UI (HTML/CSS/JS)
├── requirements.txt   # Python dependencies
└── .env.example       # Environment variable template
```

***

## Agent Details

| Property | Value |
|---|---|
| Model | `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter |
| Agent framework | OpenAI Agents SDK |
| Memory | LangChain `ConversationSummaryBufferMemory` (1000 token limit) |
| Summary LLM | `inclusionai/ling-2.6-flash:free` via OpenRouter |
| Sentiment engine | VADER (free, offline, no API needed) |
| Web search | DuckDuckGo Search (no API key required) |


***

## Memory

Valid8 maintains conversation context across turns using LangChain's `ConversationSummaryBufferMemory`. This means:

- Follow-up questions preserve the original idea context
- The agent refines searches rather than restarting from scratch
- Prior findings are compared against new evidence automatically

Useful follow-up patterns:

- *"What if I narrow the audience to B2B teams only?"*
- *"How saturated is the competitor landscape?"*
- *"Which pain point has the strongest monetization signal?"*

***

## Limitations

- Reddit and Hacker News skew toward online, tech-forward, and vocal users
- VADER sentiment can miss sarcasm and nuanced negativity
- Absence of evidence is not evidence of no demand — niche ideas may underperform in search
- High engagement does not automatically mean willingness to pay
- Competitor research focuses on app/software products — non-digital competitors require `web_search`

***
