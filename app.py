# app.py
import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import Runner, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent

# Import everything already built in z1.py — nothing changes there
from valid8 import agent, memory, langchain_memory_to_openai_format

# Remove LangChain deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ── Lifespan (startup/shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # add any startup/teardown logic here if needed

app = FastAPI(title="Market Research Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ── Human-readable tool name mapping ─────────────────────────────────────────

TOOL_LABELS = {
    "analyze_reddit":              "Scanning Reddit discussions",
    "hackernews_market_research":  "Checking Hacker News",
    "competitor_research":         "Researching competitors",
    "web_search":                  "Searching the web",
}


# ── SSE helper ────────────────────────────────────────────────────────────────

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Main chat endpoint ────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    history = langchain_memory_to_openai_format(memory, req.message)

    async def event_stream():
        final_output = ""
        tool_start_times: dict[str, float] = {}

        try:
            result = Runner.run_streamed(agent, history, max_turns=10)

            async for event in result.stream_events():

                # ── High-level item events (tool calls, final message) ──────
                if event.type == "run_item_stream_event":
                    item = event.item

                    if item.type == "tool_call_item":
                        tool_name = item.raw_item.name
                        label = TOOL_LABELS.get(tool_name, tool_name)
                        tool_start_times[tool_name] = time.perf_counter()
                        yield sse({
                            "type": "tool_start",
                            "tool": tool_name,
                            "label": label,
                        })

                    elif item.type == "tool_call_output_item":
                        # Match back to the tool that just finished
                        # The output item carries the call_id; we track by
                        # checking start times in order (tools run sequentially)
                        if tool_start_times:
                            tool_name, start = next(
                                ((k, v) for k, v in tool_start_times.items()),
                                (None, None)
                            )
                            elapsed = round(time.perf_counter() - start, 1) if start else 0
                            tool_start_times.pop(tool_name, None)
                        else:
                            tool_name, elapsed = "unknown_tool", 0

                        yield sse({
                            "type": "tool_done",
                            "tool": tool_name,
                            "elapsed": elapsed,
                        })

                    elif item.type == "message_output_item":
                        # Full final message — use this to save to memory
                        final_output = ItemHelpers.text_message_output(item)

                # ── Low-level token events (stream answer word by word) ─────
                elif event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        yield sse({
                            "type": "token",
                            "delta": event.data.delta,
                        })

            # ── Done — save to LangChain memory with retry ─────────────────
            if final_output:
                for attempt in range(4):
                    try:
                        memory.save_context(
                            {"input": req.message},
                            {"output": final_output}
                        )
                        break
                    except Exception as e:
                        if "429" in str(e) or "rate" in str(e).lower():
                            wait = 2 ** attempt
                            await asyncio.sleep(wait)
                        else:
                            break

            yield sse({"type": "done"})

        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # important: disables Nginx buffering
        },
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}