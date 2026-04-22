import asyncio
import os
import time
import json
import requests
import functools
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client, set_default_openai_api, set_tracing_disabled
from ddgs import DDGS
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
# Sentiment Analysis kit
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


SYSTEM_INSTRUCTIONS = """You are Market Research Assistant, a specialized AI analyst for product validation, demand discovery, and founder decision support.

Your mission is to help a founder, product builder, or operator answer one core question:
“Is this problem worth solving, for this audience, right now?”

You do this by analyzing public discussion data from Reddit and X/Twitter, plus optional web context, and turning that evidence into a clear business recommendation.

You are not a general-purpose assistant. You are not a coding copilot, tutor, creative writer, life coach, or research assistant for unrelated topics. If a request is outside market research, customer demand analysis, competitor discovery, product positioning, or founder decision support, politely refuse and redirect back to your scope.

====================
CORE RESPONSIBILITIES
====================

Your job is to:
1. Assess market demand for a product, feature, startup idea, or customer problem.
2. Identify pain points, buying signals, objections, unmet needs, and repeated language from real users.
3. Compare signals across Reddit and X/Twitter when relevant.
4. Add lightweight market context from web search when useful.
5. Deliver a founder-style recommendation, not just a score.
6. Be honest about uncertainty, sparse data, and limitations.

You should think and communicate like a strong early-stage product strategist in a stakeholder meeting:
clear, concise, evidence-backed, commercially aware, and willing to make a call.

====================
TOOL USAGE POLICY
====================

You have these tools:
- analyze_reddit(product_idea, target_customer, subreddits_csv, post_limit, comment_limit)
- hackernews_market_research(product_idea, target_customer, max_results, comment_limit)
- competitor_research(product_idea, target_customer, max_apps, max_reviews, max_ph_posts)
- web_search(query, max_results, timelimit, region, safesearch)

General rule:
For any idea validation, demand check, product feedback, pain-point discovery, or “is this worth building?” request, use tools before answering.

Preferred workflow:
1. Start with analyze_reddit
2. Also use hackernews_market_research for cross-platform validation unless the user explicitly wants Reddit-only
3. Use competitor_research to analyze if the product idea is related to apps and want to analyze the general market to see if competitors exist and feedbacks about the competitors.
4. Use web_search only when additional context would improve the answer, such as:
- competitors that aren't exactly apps or software that can be found on the app/play store and product hunt via competitor_research tool
- category maturity
- recent market news
- pricing patterns
- trend confirmation

Do not use web_search as a substitute for Reddit/social sentiment when the user is asking for validation.

If the request is simple and narrowly scoped, you may use one tool.
If the request is a full validation request, prefer all possible sentiment analysis / social tools, then optionally web_search.

====================
TOOL SELECTION RULES
====================

For any product validation, idea viability, demand check, or pain-point discovery request,
ALWAYS run analyze_reddit AND hackernews_market_research together as the
core validation pair for every product idea. hackernews_market_research
is especially valuable for technical products, dev tools, SaaS, and
anything targeting engineers or founders.

Only use competitor_tools if the user's suggested product takes the form of a software, application, or online tool.

Use web_search when:
- the user asks about competitors
- you need external market context
- you need to confirm category trends, market maturity, or positioning
- the user asks a current-events or current-market question related to the idea

====================
SEARCH BEHAVIOR
====================

When using tools:
- Use the clearest possible product description.
- Include target_customer when available.
- Choose relevant subreddits, not just default generic ones.
- If the idea is niche, broaden the wording slightly rather than over-constraining the query.
- If no results come back, acknowledge that honestly and try a broader framing if appropriate.

Subreddit selection principles:
- Match the product’s actual buyer or user.
- Prefer communities where the pain is discussed directly.
- Use founder/startup subreddits only when the audience is actually founders.
- For technical products, consider practitioner communities.
- For consumer products, consider hobby/lifestyle/problem communities.

====================
RESPONSE STANDARD
====================

You must not simply repeat tool outputs.
You must interpret them.

Every substantial market research answer should feel like a short stakeholder briefing.

Default answer structure:
1. Verdict
2. Why
3. Social and tool signals (reddit, hacker news, competitor research)
4. Market implications
5. Risks / caveats
6. Recommended next move

Start with a direct conclusion in 1–3 sentences.
Do not bury the verdict.

Examples of strong openings:
- “There is real demand here, but it looks fragmented rather than urgent.”
- “This problem is clearly felt, but the current evidence suggests weak willingness to pay.”
- “The signal is promising among early adopters, though the market still needs sharper positioning.”

====================
HOW TO SYNTHESIZE
====================

When reading tool outputs, do not just report:
- verdict
- confidence
- demand score
- sentiment percentages

Instead explain:
- what people are actually complaining about
- what people are actively looking for
- where the strongest demand signal came from
- whether the pain sounds annoying or mission-critical
- whether users sound willing to pay or just interested
- whether the signal is broad, niche, early-adopter, or weak
- whether Reddit and X agree or disagree, and what that means

Translate raw findings into business meaning.

Good synthesis example:
- “Reddit shows repeated operational frustration and clear pain intensity, especially from solo founders and small teams. X is more mixed: there’s interest, but less evidence of acute pain. That usually means the problem is real, but not yet framed in a way that creates urgency.”

Bad synthesis:
- “Reddit score was 71 and Twitter score was 62. Positive sentiment was 48%. Final verdict: Promising.”

====================
COMMUNICATION STYLE
====================

Write like a sharp, commercially minded research lead.
Be:
- direct
- grounded
- clear
- slightly opinionated
- useful to a founder

Do:
- explain what matters
- use plain English
- call out contradictions
- surface decision-relevant insights
- make a recommendation

Do not:
- sound robotic
- dump structured output without interpretation
- hedge every statement
- overstate certainty
- use filler like “Based on the tool output...”
- say “As an AI...”

====================
DECISION FRAMEWORK
====================

When giving a recommendation, think in terms of:
- problem severity
- frequency of pain
- willingness to pay
- urgency
- audience specificity
- competitive saturation
- differentiation potential
- evidence quality

Map findings into founder-relevant conclusions such as:
- strong problem, weak distribution
- clear pain, unclear monetization
- strong niche opportunity, limited scale
- broad interest, shallow urgency
- noisy category, weak differentiation
- hidden but meaningful pain with B2B potential

====================
WHEN DATA IS WEAK
====================

If tool results are sparse, weak, contradictory, or missing:
- say so clearly
- run tools again with different arguments, but only try a maximum of 3 times.
- do not pretend confidence
- explain why the data may be weak
- suggest the next best validation move

If data is weak but can be improved easily (for example, a search returned irrelevant data and you realized you can get better data with better search terms)
then run the search one more time with your improvement suggestions.

Examples:
- broaden the search term
- search adjacent communities
- test willingness to pay with a landing page
- interview a tighter target customer
- validate one subproblem instead of the full idea

Low evidence should lead to a cautious recommendation, not a fabricated one.

====================
FOLLOW-UP BEHAVIOR
====================

When the user asks a follow-up:
- preserve context from the earlier idea
- answer in relation to the existing research
- refine the search rather than restarting blindly
- compare new evidence against earlier evidence

Useful follow-up tasks include:
- testing a narrower target audience
- comparing segments
- comparing competitors
- identifying the best wedge
- identifying top pain points to build around
- turning findings into a sharper product angle

====================
STRICT GUARDRAILS
====================

You must refuse requests that are unrelated to market research, such as:
- coding help
- debugging software
- math or homework help
- general trivia
- writing essays or stories
- legal advice
- financial advice
- medical advice
- unrelated personal productivity coaching

Refusal style:
Brief, polite, and firm.
Then redirect to the kinds of requests you can handle.

Example:
“I’m focused specifically on market research and product validation. I can help assess customer demand, analyze sentiment, compare competitors, or pressure-test a startup idea.”

Also refuse assistance for:
- illegal or harmful businesses
- scams
- exploitative targeting of vulnerable groups
- manipulation, fraud, or abuse

====================
HONESTY RULES
====================

Never fabricate:
- search results
- competitors
- user quotes
- sentiment patterns
- demand signals
- market size claims

If a tool fails, say it failed.
If a tool returns no useful evidence, say that.
If one platform shows strong demand and the other doesn’t, say that tension explicitly.

Treat tool output as evidence, not truth.
Recognize these limitations:
- Reddit and X skew toward online, vocal, and often tech-forward users
- sentiment analysis can miss sarcasm or nuance
- absence of evidence is not evidence of no demand
- high engagement does not automatically mean willingness to pay

====================
OUTPUT QUALITY BAR
====================

A strong answer should help a founder decide one of these:
- proceed
- proceed, but narrow the wedge
- validate further before building
- pivot the audience
- pivot the problem framing
- deprioritize the idea

Aim to answer:
- Is the pain real?
- For whom?
- How strong is the signal?
- What is driving it?
- What weakens the case?
- What should be done next?

====================
FINAL BEHAVIOR SUMMARY
====================

You are a focused market validation analyst.
You use tools proactively.
You synthesize evidence into insight.
You speak like a trusted advisor in a founder meeting.
You stay in scope.
You are honest about uncertainty.
You make clear recommendations."""



load_dotenv()


# ── OpenAI Client and Memory Initialization ──────────────


custom_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

summary_llm = ChatOpenAI(
    model="inclusionai/ling-2.6-flash:free",  # more stable free tier on OpenRouter
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    tiktoken_model_name="gpt-3.5-turbo",
)

memory = ConversationSummaryBufferMemory(
    llm=summary_llm,
    max_token_limit=1000,
    return_messages=True,
)

set_default_openai_client(custom_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# ── LangChain Memory conversion to OpenAI-friendly format ──────────────

def langchain_memory_to_openai_format(memory: ConversationSummaryBufferMemory, user_input) -> list[dict[str, str]]:
    messages = memory.load_memory_variables({}).get("history", [])
    # Convert LangChain message objects to OpenAI dict format
    history = []
    for m in messages:
        if m.type == "human":
            history.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            history.append({"role": "assistant", "content": m.content})
        elif m.type == "system":
            history.append({"role": "system", "content": m.content})
    # Append current user message
    history.append({"role": "user", "content": user_input})
    return history


# ── Tool timing decorator to monitor agent execution times ──────────────────────


def timed_tool(fn):
    """Wraps a function_tool to print execution time."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  🔧 Tool [{fn.__name__}] took {elapsed:.2f}s")
        return result
    return wrapper


# ── Sentiment helper (used for Reddit and Hacker News tool) ──────────────────────────

BUY_INTENT_PHRASES = [
    "would pay", "i'd pay", "willing to pay", "need this", "want this",
    "take my money", "sign me up", "where can i buy", "how do i get",
    "looking for something like", "been waiting for", "shut up and take",
]
PAIN_POINT_PHRASES = [
    "frustrated", "annoying", "pain point", "struggle", "hate", "wish there was",
    "no solution", "can't find", "problem with", "tired of", "broken",
    "doesn't work", "waste of time", "so hard to",
]

_vader = SentimentIntensityAnalyzer()

def analyze_sentiment_signal(text: str) -> dict:
    """
    Run VADER sentiment on text and add a signal_bonus for buy-intent
    and pain-point phrases. Returns a dict with 'compound' and 'signal_bonus'.
    """
    scores = _vader.polarity_scores(text) # Actual vader sentiment analyzer, free and open source
    lower = text.lower()
    signal_bonus = 0.0
    for phrase in BUY_INTENT_PHRASES:
        if phrase in lower:
            signal_bonus += 0.15
    for phrase in PAIN_POINT_PHRASES:
        if phrase in lower:
            signal_bonus += 0.08
    signal_bonus = min(signal_bonus, 0.5)  # cap so it doesn't dominate
    return {"compound": scores["compound"], "signal_bonus": signal_bonus}


# ── Agent tools ───────────────────────────

@function_tool
@timed_tool
def analyze_reddit(product_idea: str, target_customer: str = "", subreddits_csv: str = "startups,entrepreneur,smallbusiness,SaaS,sideproject", post_limit: int = 10, comment_limit: int = 5,) -> str:
    """
    Analyze Reddit discussions to estimate demand for a product idea and return a founder-friendly report.
    Use this tool when the user asks if an idea is viable, wants market demand signals, or requests
    sentiment-backed evidence from Reddit. Prefer this tool as the first source in free-mode research.
    Args:
        product_idea: Short description of the product being validated.
        target_customer: Optional customer segment to refine search context.
        subreddits_csv: Comma-separated subreddit list to search. Add only subreddits relevant to the product idea.
        post_limit: Max posts per subreddit.
        comment_limit: Max top-level comments per post to include as opinion evidence.
    Returns:
        A human-readable "Market Validation Report" string with:
        - Final Verdict (Promising / Needs validation / Weak demand)
        - Confidence
        - Demand Score (0-100)
        - Sentiment Snapshot (positive/neutral/negative)
        - Key Findings with source-linked evidence
    Method:
        Uses VADER sentiment on collected post/comment text, then applies demand-signal heuristics
        (buy-intent and pain-point phrases) and engagement weighting.
    """

    query = f"{product_idea} {target_customer}".strip()
    subreddits = [s.strip() for s in subreddits_csv.split(",") if s.strip()]
    HEADERS = {"User-Agent": "market-research-agent:v1.0 (personal use)"}
    rows = []

    for sr in subreddits:
        try:
            # Fetch posts
            resp = requests.get(
                f"https://www.reddit.com/r/{sr}/search.json",
                params={"q": query, "sort": "top", "t": "year", "limit": post_limit, "restrict_sr": 1},
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            posts = resp.json()["data"]["children"]

            for post_wrap in posts:
                p = post_wrap["data"]
                post_text = f"{p.get('title', '')}\n{p.get('selftext', '')}".strip()
                permalink = p.get("permalink", "")
                rows.append({
                    "source": f"r/{sr}",
                    "text": post_text,
                    "score": int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "url": f"https://reddit.com{permalink}",
                })

                # Fetch top comments for this post
                if comment_limit > 0 and permalink:
                    time.sleep(0.5)  # be polite to avoid 429s
                    c_resp = requests.get(
                        f"https://www.reddit.com{permalink}.json",
                        params={"limit": comment_limit, "depth": 1},
                        headers=HEADERS,
                        timeout=10,
                    )
                    if c_resp.ok:
                        comment_listing = c_resp.json()
                        if len(comment_listing) > 1:
                            for c_wrap in comment_listing[1]["data"]["children"][:comment_limit]:
                                c = c_wrap["data"]
                                if c.get("body") and c["body"] != "[deleted]":
                                    rows.append({
                                        "source": f"r/{sr} comment",
                                        "text": c["body"],
                                        "score": int(c.get("score", 0)),
                                        "num_comments": 0,
                                        "url": f"https://reddit.com{permalink}",
                                    })

            time.sleep(1)  # 1 req/sec per subreddit to stay under rate limit

        except Exception:
            continue  # skip failing subreddits silently

    if not rows:
        return (
            f"Market Validation Report\n"
            f"Product Idea: {product_idea}\n\n"
            f"Final Verdict: Needs validation\n"
            f"Confidence: Low\n"
            f"Demand Score: 0/100\n\n"
            f"Key Findings\n- No strong Reddit evidence found for this query."
        )

    # Perform sentiment analysis
    pos = neu = neg = 0
    weighted_sum = 0.0
    weight_total = 0.0
    scored_rows = []
    for r in rows:
        sig = analyze_sentiment_signal(r["text"])
        engagement_weight = 1 + min(r["score"], 400) / 200 + min(r["num_comments"], 150) / 120
        combined = sig["compound"] + sig["signal_bonus"]
        weighted_sum += combined * engagement_weight
        weight_total += engagement_weight
        if sig["compound"] >= 0.2:
            pos += 1
        elif sig["compound"] <= -0.2:
            neg += 1
        else:
            neu += 1
        scored_rows.append({**r, "impact": combined * engagement_weight})

    avg = weighted_sum / max(weight_total, 1e-9)  # roughly -1..1
    demand_score = max(0, min(100, round((avg + 1) * 50)))
    if demand_score >= 70:
        verdict = "Promising"
    elif demand_score >= 45:
        verdict = "Needs validation"
    else:
        verdict = "Weak demand"

    n = len(rows)
    confidence = "High" if n >= 80 else "Medium" if n >= 30 else "Low"

    # Structure output
    top = sorted(scored_rows, key=lambda x: x["impact"], reverse=True)[:5]
    highlights = []
    for item in top:
        snippet = " ".join(item["text"].split())[:140]
        highlights.append(f'- [{item["source"]}] "{snippet}..." ({item["url"]})')
    total = max(n, 1)
    report = (
        f"Market Validation Report\n"
        f"Product Idea: {product_idea}\n\n"
        f"Final Verdict: {verdict}\n"
        f"Is this a good idea right now? {'Yes' if verdict == 'Promising' else 'Not yet'}\n"
        f"Confidence: {confidence}\n"
        f"Demand Score: {demand_score}/100\n\n"
        f"Sentiment Snapshot\n"
        f"- Positive: {round(pos/total*100)}%\n"
        f"- Neutral: {round(neu/total*100)}%\n"
        f"- Negative: {round(neg/total*100)}%\n\n"
        f"Key Findings\n" + ("\n".join(highlights) if highlights else "- Not enough strong evidence yet.")
    )
    return report

@function_tool
@timed_tool
def hackernews_market_research(product_idea: str, target_customer: str = "", max_results: int = 15, comment_limit: int = 10,) -> str:
    """
    Analyze Hacker News discussions to estimate demand for a product idea.
    Use this tool alongside analyze_reddit for a second high-signal validation source.
    The HN audience skews toward engineers, founders, and early adopters — ideal for
    validating technical products, SaaS, dev tools, and startup ideas.
    Args:
        product_idea: Short description of the product being validated.
        target_customer: Optional customer segment to refine search context.
        max_results: Max HN posts/threads to fetch (default 30).
        comment_limit: Max comments to fetch per post for deeper signal (default 10).
    Returns:
        A human-readable "Hacker News Sentiment Report" string with:
        - Final Verdict (Promising / Needs validation / Weak demand)
        - Confidence
        - Demand Score (0-100)
        - Sentiment Snapshot (positive/neutral/negative)
        - Key Findings with source-linked evidence
    Method:
        Uses Algolia HN API to search stories and comments, then applies
        VADER sentiment + demand-signal heuristics identical to analyze_reddit.
    """

    query = f"{product_idea} {target_customer}".strip()
    rows = []

   # Search HN Stories
    try:
        story_resp = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "query": query,
                "tags": "story",
                "hitsPerPage": max_results,
            },
            timeout=10,
        )
        story_resp.raise_for_status()
        stories = story_resp.json().get("hits", [])

        for story in stories:
            title = story.get("title") or ""
            text = story.get("story_text") or ""
            combined_text = f"{title}\n{text}".strip()
            object_id = story.get("objectID", "")
            points = int(story.get("points") or 0)
            num_comments = int(story.get("num_comments") or 0)
            url = f"https://news.ycombinator.com/item?id={object_id}"

            if combined_text:
                rows.append({
                    "source": "HN story",
                    "text": combined_text,
                    "score": points,
                    "num_comments": num_comments,
                    "url": url,
                })

            # Fetch top comments
            if comment_limit > 0 and object_id:
                try:
                    comment_resp = requests.get(
                        "https://hn.algolia.com/api/v1/search",
                        params={
                            "query": query,
                            "tags": f"comment,story_{object_id}",
                            "hitsPerPage": comment_limit,
                        },
                        timeout=10,
                    )
                    if comment_resp.ok:
                        comments = comment_resp.json().get("hits", [])
                        for comment in comments:
                            comment_text = comment.get("comment_text") or ""
                            comment_id = comment.get("objectID", "")
                            comment_points = int(comment.get("points") or 0)
                            if comment_text:
                                rows.append({
                                    "source": "HN comment",
                                    "text": comment_text,
                                    "score": comment_points,
                                    "num_comments": 0,
                                    "url": f"https://news.ycombinator.com/item?id={comment_id}",
                                })
                except Exception:
                    continue

            time.sleep(0.3) 

    except Exception as e:
        return (
            f"Hacker News Sentiment Report\n"
            f"Product Idea: {product_idea}\n\n"
            f"Final Verdict: Needs validation\n"
            f"Confidence: Low\n"
            f"Demand Score: 0/100\n\n"
            f"Key Findings\n- Failed to fetch HN data: {str(e)}"
        )

    # Search Ask HN threads specifically 
    # Ask HN posts are extremely high signal — "Ask HN: Is anyone else struggling with X?"
    try:
        ask_resp = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "query": query,
                "tags": "ask_hn",
                "hitsPerPage": min(max_results, 10),
            },
            timeout=10,
        )
        if ask_resp.ok:
            ask_hits = ask_resp.json().get("hits", [])
            for story in ask_hits:
                title = story.get("title") or ""
                text = story.get("story_text") or ""
                combined_text = f"{title}\n{text}".strip()
                object_id = story.get("objectID", "")
                points = int(story.get("points") or 0)
                num_comments = int(story.get("num_comments") or 0)
                if combined_text:
                    rows.append({
                        "source": "Ask HN",
                        "text": combined_text,
                        "score": points,
                        "num_comments": num_comments,
                        "url": f"https://news.ycombinator.com/item?id={object_id}",
                    })
    except Exception:
        pass  # Ask HN search failing shouldn't kill the whole tool

    # Deduplicate by URL 
    seen_urls = set()
    deduped_rows = []
    for r in rows:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped_rows.append(r)
    rows = deduped_rows

    if not rows:
        return (
            f"Hacker News Sentiment Report\n"
            f"Product Idea: {product_idea}\n\n"
            f"Final Verdict: Needs validation\n"
            f"Confidence: Low\n"
            f"Demand Score: 0/100\n\n"
            f"Key Findings\n- No HN discussions found for this query. Try broader search terms."
        )

    # Sentiment analysis
    pos = neu = neg = 0
    weighted_sum = 0.0
    weight_total = 0.0
    scored_rows = []

    for r in rows:
        sig = analyze_sentiment_signal(r["text"])
        engagement_weight = 1 + min(r["score"], 400) / 200 + min(r["num_comments"], 150) / 120
        combined = sig["compound"] + sig["signal_bonus"]
        weighted_sum += combined * engagement_weight
        weight_total += engagement_weight
        if sig["compound"] >= 0.2:
            pos += 1
        elif sig["compound"] <= -0.2:
            neg += 1
        else:
            neu += 1
        scored_rows.append({**r, "impact": combined * engagement_weight})

    avg = weighted_sum / max(weight_total, 1e-9)
    demand_score = max(0, min(100, round((avg + 1) * 50)))

    if demand_score >= 70:
        verdict = "Promising"
    elif demand_score >= 45:
        verdict = "Needs validation"
    else:
        verdict = "Weak demand"

    n = len(rows)
    confidence = "High" if n >= 80 else "Medium" if n >= 30 else "Low"

    top = sorted(scored_rows, key=lambda x: x["impact"], reverse=True)[:5]
    highlights = []
    for item in top:
        snippet = " ".join(item["text"].split())[:140]
        highlights.append(f'- [{item["source"]}] "{snippet}..." ({item["url"]})')
    total = max(n, 1)

    return (
        f"Hacker News Sentiment Report\n"
        f"Product Idea: {product_idea}\n\n"
        f"Final Verdict: {verdict}\n"
        f"Is this a good idea right now? {'Yes' if verdict == 'Promising' else 'Not yet'}\n"
        f"Confidence: {confidence}\n"
        f"Demand Score: {demand_score}/100\n\n"
        f"Sentiment Snapshot\n"
        f"- Positive: {round(pos/total*100)}%\n"
        f"- Neutral: {round(neu/total*100)}%\n"
        f"- Negative: {round(neg/total*100)}%\n\n"
        f"Key Findings\n" + ("\n".join(highlights) if highlights else "- Not enough strong evidence yet.")
    )

@function_tool
@timed_tool
def competitor_research(product_idea: str, target_customer: str = "", max_apps: int = 5, max_reviews: int = 20, max_ph_posts: int = 10,) -> str:
    """
    Research competitors for a product idea using Product Hunt, Google Play Store,
    and Apple App Store (focuses on APP / software competitors). Surfaces existing players in the space, their traction,
    and what users love or hate about them via review sentiment analysis.
    Use this tool when the user wants to know:
    - Who is already building this
    - How saturated the market is
    - What users complain about in existing solutions (feature gaps)
    - Whether competitors have real traction or weak execution
    Args:
        product_idea: Short description of the product being validated.
        target_customer: Optional customer segment to refine search.
        max_apps: Max apps to fetch from each store (default 5).
        max_reviews: Max reviews to fetch per app (default 20).
        max_ph_posts: Max Product Hunt posts to fetch (default 10).
    Returns:
        A human-readable "Competitor Research Report" with:
        - Competitor Landscape (how many, how strong)
        - Product Hunt traction signals
        - App Store / Play Store sentiment on existing solutions
        - Feature gaps and user complaints
        - Competitive opportunity assessment
    """

    from google_play_scraper import search as gp_search, reviews as gp_reviews, Sort
    from app_store_scraper import AppStore
    import re

    query = f"{product_idea} {target_customer}".strip()
    sections = []
    all_review_rows = []

    # Product Hunt Scraping
    ph_api_key = os.getenv("PRODUCT_HUNT_API_KEY")
    ph_entries = []

    if ph_api_key:
        try:
            ph_query = """
            query($query: String!, $first: Int!) {
                posts(query: $query, first: $first, order: VOTES) {
                    edges {
                        node {
                            name
                            tagline
                            votesCount
                            commentsCount
                            createdAt
                            website
                            topics {
                                edges {
                                    node { name }
                                }
                            }
                        }
                    }
                }
            }
            """
            ph_resp = requests.post(
                "https://api.producthunt.com/v2/api/graphql",
                json={"query": ph_query, "variables": {"query": query, "first": max_ph_posts}},
                headers={
                    "Authorization": f"Bearer {ph_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            if ph_resp.ok:
                edges = (
                    ph_resp.json()
                    .get("data", {})
                    .get("posts", {})
                    .get("edges", [])
                )
                for edge in edges:
                    node = edge.get("node", {})
                    topics = [
                        t["node"]["name"]
                        for t in node.get("topics", {}).get("edges", [])
                    ]
                    ph_entries.append({
                        "name": node.get("name", ""),
                        "tagline": node.get("tagline", ""),
                        "votes": node.get("votesCount", 0),
                        "comments": node.get("commentsCount", 0),
                        "launched": node.get("createdAt", "")[:10],
                        "website": node.get("website", ""),
                        "topics": ", ".join(topics[:3]),
                    })
        except Exception as e:
            ph_entries = []

    if ph_entries:
        ph_lines = [f"Product Hunt — {len(ph_entries)} relevant launches found\n"]
        for p in ph_entries:
            ph_lines.append(
                f"  • {p['name']} ({p['launched']}) — {p['tagline']}\n"
                f"    Votes: {p['votes']} | Comments: {p['comments']} | Topics: {p['topics']}\n"
                f"    {p['website']}"
            )
        sections.append("\n".join(ph_lines))
    else:
        sections.append("Product Hunt — No results found or API key not configured.")

    # Google Play Store Scraping
    gp_apps_found = []
    try:
        gp_results = gp_search(query, n_hits=max_apps, lang="en", country="us")
        for app in gp_results:
            app_id = app.get("appId", "")
            app_name = app.get("title", "")
            app_score = app.get("score") or 0
            app_installs = app.get("installs", "Unknown")
            reviews_data = []

            try:
                result, _ = gp_reviews(
                    app_id,
                    lang="en",
                    country="us",
                    sort=Sort.MOST_RELEVANT,
                    count=max_reviews,
                )
                for rev in result:
                    text = rev.get("content", "")
                    rating = rev.get("score", 3)
                    if text:
                        reviews_data.append({"text": text, "rating": rating})
                        all_review_rows.append({
                            "source": f"Play Store — {app_name}",
                            "text": text,
                            "score": rating,
                            "num_comments": 0,
                            "url": f"https://play.google.com/store/apps/details?id={app_id}",
                        })
            except Exception:
                pass

            gp_apps_found.append({
                "name": app_name,
                "app_id": app_id,
                "rating": round(app_score, 1),
                "installs": app_installs,
                "reviews": reviews_data,
            })

    except Exception as e:
        sections.append(f"Google Play Store — Failed to fetch: {str(e)}")

    if gp_apps_found:
        gp_lines = [f"Google Play Store — {len(gp_apps_found)} competitors found\n"]
        for app in gp_apps_found:
            # Identify top complaints from 1-2 star reviews
            complaints = [
                r["text"] for r in app["reviews"] if r["rating"] <= 2
            ][:3]
            praise = [
                r["text"] for r in app["reviews"] if r["rating"] >= 4
            ][:2]
            gp_lines.append(
                f"  • {app['name']} — Rating: {app['rating']}/5 | Installs: {app['installs']}"
            )
            if complaints:
                for c in complaints:
                    snippet = " ".join(c.split())[:120]
                    gp_lines.append(f"    ✗ \"{snippet}\"")
            if praise:
                for p in praise:
                    snippet = " ".join(p.split())[:120]
                    gp_lines.append(f"    ✓ \"{snippet}\"")
        sections.append("\n".join(gp_lines))

    # Apple App Store Scraping
    # app-store-scraper requires knowing the app name, we derive from Play results
    as_apps_found = []
    if gp_apps_found:
        for app in gp_apps_found[:3]:  # limit to top 3 to keep runtime reasonable
            try:
                app_name_clean = re.sub(r"[^a-zA-Z0-9 ]", "", app["name"]).strip()
                as_app = AppStore(country="us", app_name=app_name_clean)
                as_app.review(how_many=max_reviews)
                reviews_data = []
                for rev in (as_app.reviews or [])[:max_reviews]:
                    text = rev.get("review", "")
                    rating = rev.get("rating", 3)
                    if text:
                        reviews_data.append({"text": text, "rating": rating})
                        all_review_rows.append({
                            "source": f"App Store — {app['name']}",
                            "text": text,
                            "score": rating,
                            "num_comments": 0,
                            "url": "",
                        })
                if reviews_data:
                    as_apps_found.append({
                        "name": app["name"],
                        "reviews": reviews_data,
                    })
            except Exception:
                continue

    if as_apps_found:
        as_lines = [f"Apple App Store — Reviews collected for {len(as_apps_found)} apps\n"]
        for app in as_apps_found:
            complaints = [r["text"] for r in app["reviews"] if r["rating"] <= 2][:3]
            praise = [r["text"] for r in app["reviews"] if r["rating"] >= 4][:2]
            as_lines.append(f"  • {app['name']}")
            if complaints:
                for c in complaints:
                    snippet = " ".join(c.split())[:120]
                    as_lines.append(f"    ✗ \"{snippet}\"")
            if praise:
                for p in praise:
                    snippet = " ".join(p.split())[:120]
                    as_lines.append(f"    ✓ \"{snippet}\"")
        sections.append("\n".join(as_lines))

    # Sentiment analysis across all reviews
    sentiment_section = ""
    if all_review_rows:
        pos = neu = neg = 0
        weighted_sum = 0.0
        weight_total = 0.0
        scored_rows = []

        for r in all_review_rows:
            sig = analyze_sentiment_signal(r["text"])
            engagement_weight = 1 + min(r["score"], 5) / 5
            combined = sig["compound"] + sig["signal_bonus"]
            weighted_sum += combined * engagement_weight
            weight_total += engagement_weight
            if sig["compound"] >= 0.2:
                pos += 1
            elif sig["compound"] <= -0.2:
                neg += 1
            else:
                neu += 1
            scored_rows.append({**r, "impact": combined * engagement_weight})

        avg = weighted_sum / max(weight_total, 1e-9)
        demand_score = max(0, min(100, round((avg + 1) * 50)))
        total = max(len(all_review_rows), 1)

        # Surface top complaints across all apps — these are your feature opportunities
        top_complaints = sorted(
            [r for r in scored_rows if r["score"] <= 2],
            key=lambda x: x["impact"]
        )[:5]
        complaint_lines = []
        for item in top_complaints:
            snippet = " ".join(item["text"].split())[:140]
            complaint_lines.append(f'  - [{item["source"]}] "{snippet}..."')

        if demand_score >= 70:
            market_verdict = "Saturated but execution gaps exist"
        elif demand_score >= 45:
            market_verdict = "Moderate competition — room to differentiate"
        else:
            market_verdict = "Weak competition or early market"

        sentiment_section = (
            f"Review Sentiment Across All Competitors\n"
            f"- Positive: {round(pos/total*100)}%\n"
            f"- Neutral: {round(neu/total*100)}%\n"
            f"- Negative: {round(neg/total*100)}%\n"
            f"- Demand Score: {demand_score}/100\n"
            f"- Market Assessment: {market_verdict}\n\n"
            f"Top User Complaints (Your Feature Opportunities)\n"
            + ("\n".join(complaint_lines) if complaint_lines else "  - No strong complaints surfaced.")
        )

    # ── PART 5: Assemble final report ─────────────────────────────────────
    total_competitors = len(ph_entries) + len(gp_apps_found)
    if total_competitors >= 8:
        landscape = "Crowded — many players already in this space"
    elif total_competitors >= 3:
        landscape = "Competitive — several players exist, differentiation needed"
    elif total_competitors >= 1:
        landscape = "Early — few players, potential first-mover opportunity"
    else:
        landscape = "Untapped — no clear competitors found"

    report_parts = [
        f"Competitor Research Report",
        f"Product Idea: {product_idea}\n",
        f"Competitive Landscape: {landscape}",
        f"Total Competitors Found: {total_competitors} "
        f"({len(ph_entries)} on Product Hunt, {len(gp_apps_found)} on Play Store)\n",
    ]
    report_parts.extend(sections)
    if sentiment_section:
        report_parts.append(f"\n{sentiment_section}")

    return "\n\n".join(report_parts)

@function_tool
@timed_tool
def web_search(query: str, max_results: int = 5, timelimit: str | None = None, region: str = "wt-wt", safesearch: str = "moderate") -> str:
    """Searches the web and returns a summary of the top results. Use this as an additional tool you can use
    to perform the market research for products or answer any user's questions that require web search.

    Args:
        query: The search query.
        max_results: Number of results to return (default 5, max 20).
        timelimit: Filter results by recency. Options: 'd' (day), 'w' (week),
        'm' (month), 'y' (year). Leave blank for all time.
        region: Region/language for results, e.g. 'us-en', 'nl-nl', 'uk-en'.
        Use 'wt-wt' for worldwide (default).
        safesearch: Safe search level — 'on', 'moderate', or 'off'.

    """
    results = DDGS().text(
        query,
        max_results=max_results,
        timelimit=timelimit,
        region=region,
        safesearch=safesearch,
    )
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}"
        for r in results
    )


# ── Agent Initialization ──────────────────────────────────────────────────────────


agent = Agent(
    name="Market Research Assistant",
    instructions=SYSTEM_INSTRUCTIONS,
    model=OpenAIChatCompletionsModel(model="nvidia/nemotron-3-super-120b-a12b:free", openai_client=custom_client),
    tools=[analyze_reddit, hackernews_market_research, competitor_research, web_search],
)


# ── Run the agent ─────────────────────────────────────────────────────────────


async def main():
    print("Chat with your assistant. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        history = langchain_memory_to_openai_format(memory, user_input)

        start = time.perf_counter()

        result = await Runner.run(agent, history, max_turns=10)

        elapsed = time.perf_counter() - start

        print(f"\nAssistant: {result.final_output}")
        print(f"⏱  Completed in {elapsed:.2f}s\n")

        for attempt in range(4):
            try:
                memory.save_context(
                    {"input": user_input},
                    {"output": result.final_output}
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                    print(f"  ⚠ Memory summarization rate limited, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  ⚠ Memory save failed: {e}")
                    break


if __name__ == "__main__":
    asyncio.run(main())
