import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamChat } from "./api";

const TOOL_LABELS = {
  thinking: "Analyzing your request...",
  analyze_reddit: "Scanning Reddit discussions",
  hackernews_market_research: "Checking Hacker News",
  competitor_research: "Researching competitors on App Store, Play Store, Product Hunt...",
  web_search: "Searching the web",
};

function ToolActivity({ steps }) {
  if (steps.length === 0) return null;
  return (
    <div className="tool-activity">
      <span className="tool-activity-title">Agent activity</span>
      {steps.map((step, i) => (
        <div key={i} className="tool-step">
          <span className={`tool-icon ${step.done ? "done" : "running"}`}>
            {step.done ? "✓" : <span className="spinner" />}
          </span>
          <span className="tool-label">{TOOL_LABELS[step.tool] ?? step.tool}</span>
          {step.elapsed && (
            <span className="tool-elapsed">{step.elapsed}s</span>
          )}
        </div>
      ))}
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={`message message-${msg.role}`}>
      {msg.role === "assistant" && (
        <ToolActivity steps={msg.toolSteps ?? []} />
      )}
      <div className="message-bubble">
        {msg.role === "assistant" ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {msg.content || (msg.streaming ? "▍" : "")}
          </ReactMarkdown>
        ) : (
          <p>{msg.content}</p>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef(null);
  const cancelRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function updateLastMessage(updater) {
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = updater(updated[updated.length - 1]);
      return updated;
    });
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setIsLoading(true);

    // Add user message + assistant placeholder with synthetic "thinking" step
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage },
      {
        role: "assistant",
        content: "",
        toolSteps: [{ tool: "thinking", done: false }],
        streaming: true,
      },
    ]);

    cancelRef.current = streamChat(userMessage, (event) => {
      if (event.type === "tool_start") {
        // Replace the synthetic "thinking" step with the real tool step
        updateLastMessage((msg) => ({
          ...msg,
          toolSteps: [
            ...msg.toolSteps.filter((s) => s.tool !== "thinking"),
            { tool: event.tool, done: false },
          ],
        }));

      } else if (event.type === "tool_done") {
        updateLastMessage((msg) => {
          const steps = [...(msg.toolSteps ?? [])];
          const idx = steps.map((s) => s.done).lastIndexOf(false);
          if (idx !== -1) steps[idx] = { ...steps[idx], done: true, elapsed: event.elapsed };
          return { ...msg, toolSteps: steps };
        });

      } else if (event.type === "token") {
        updateLastMessage((msg) => ({
          ...msg,
          content: (msg.content ?? "") + event.delta,
        }));

      } else if (event.type === "done") {
        updateLastMessage((msg) => ({ ...msg, streaming: false }));
        setIsLoading(false);

      } else if (event.type === "error") {
        updateLastMessage((msg) => ({
          ...msg,
          content: `Error: ${event.message}`,
          streaming: false,
          toolSteps: [],
        }));
        setIsLoading(false);
      }
    });
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <svg aria-label="Valid8" viewBox="0 0 32 32" fill="none" width="28" height="28">
            <rect x="2" y="2" width="28" height="28" rx="6" fill="currentColor" opacity="0.12" />
            <path d="M8 16l5 5 11-11" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span>Valid8</span>
        </div>
        <span className="header-sub">Market Research Agent</span>
      </header>

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Describe a product idea and I'll validate it using Reddit, Hacker News, and competitor data.</p>
            <p className="empty-example">Try: <em>"Is there demand for an AI-powered personal finance tracker for freelancers?"</em></p>
          </div>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        <div ref={bottomRef} />
      </main>

      <form className="input-area" onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }
          }}
          placeholder="Describe a product idea to validate..."
          rows={1}
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()} aria-label="Send">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
            <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </form>
    </div>
  );
}