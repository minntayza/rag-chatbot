import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  Bot,
  ChevronRight,
  Database,
  FileText,
  PanelRightClose,
  PanelRightOpen,
  Paperclip,
  Send,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  User,
  Zap,
} from "lucide-react";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  type ChatResponse,
  type StreamEvent,
  getHistory,
  sendMessage,
  streamMessage,
  submitFeedback,
  uploadDocument,
} from "@/lib/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "RAG Chatbot — AI Customer Assistant" },
      {
        name: "description",
        content:
          "Enterprise AI customer support console with retrieval-augmented context and transparent debugging.",
      },
      { property: "og:title", content: "RAG Chatbot — AI Customer Assistant" },
      {
        property: "og:description",
        content:
          "Enterprise AI customer support console with retrieval-augmented context and transparent debugging.",
      },
    ],
  }),
  component: Index,
});

// ── Types ───────────────────────────────────────────────────────────

type Chunk = {
  id: string;
  source: string;
  score: number;
  snippet: string;
};

type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  /** Real API metadata — present when the bot response came from the backend */
  meta?: BotMeta;
};

type BotMeta = {
  responseTimeMs: number;
  generationLatencyMs: number;
  inputTokens: number;
  outputTokens: number;
  sources: string[];
  chunks: Chunk[];
};

// ── Helpers ─────────────────────────────────────────────────────────

function apiSourcesToChunks(sources: string[]): Chunk[] {
  return sources.map((s, i) => ({
    id: `chunk-${i}`,
    source: s,
    score: 0,
    snippet: "",
  }));
}

// ── Main component ──────────────────────────────────────────────────

function Index() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeChunks, setActiveChunks] = useState<Chunk[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load chat history on mount
  useEffect(() => {
    (async () => {
      try {
        const history = await getHistory();
        if (history.messages.length > 0) {
          const msgs: Message[] = history.messages.map((m) => ({
            id: m.id,
            role: m.role === "user" ? "user" : "bot",
            content: m.message,
          }));
          setMessages(msgs);
        }
      } catch {
        // History unavailable — start fresh
      } finally {
        setLoadingHistory(false);
      }
    })();
  }, []);

  // Auto-scroll when messages change
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  // ── Non-streaming send ─────────────────────────────────
  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || streaming) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");

    try {
      const data: ChatResponse = await sendMessage(trimmed);
      const meta: BotMeta = {
        responseTimeMs: data.retrieval_latency_ms + data.generation_latency_ms,
        generationLatencyMs: data.generation_latency_ms,
        inputTokens: data.input_tokens,
        outputTokens: data.output_tokens,
        sources: data.sources,
        chunks: apiSourcesToChunks(data.sources),
      };
      const botMsg: Message = {
        id: data.id,
        role: "bot",
        content: data.message,
        meta,
      };
      setMessages((m) => [...m, botMsg]);
      setActiveChunks(meta.chunks);
    } catch (err: any) {
      toast.error(err.message || "Failed to get a response");
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "bot",
        content:
          "⚠️ Sorry, I couldn't process that request. Please check that the backend is running and try again.",
      };
      setMessages((m) => [...m, errorMsg]);
    }
  };

  // ── Streaming send ─────────────────────────────────────
  const handleSendStream = async () => {
    const trimmed = input.trim();
    if (!trimmed || streaming) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };

    // Placeholder bot message that will fill in as tokens arrive
    const placeholderId = crypto.randomUUID();
    const placeholderMsg: Message = {
      id: placeholderId,
      role: "bot",
      content: "",
    };

    setMessages((m) => [...m, userMsg, placeholderMsg]);
    setInput("");
    setStreaming(true);

    let fullContent = "";
    let finalSources: string[] = [];
    let retrievalMs = 0;

    try {
      for await (const event of streamMessage(trimmed)) {
        switch (event.type) {
          case "status":
            retrievalMs = event.retrieval_latency_ms ?? 0;
            // Could show a subtle "retrieving..." indicator here
            break;
          case "token":
            fullContent += event.token ?? "";
            setMessages((prev) =>
              prev.map((m) =>
                m.id === placeholderId ? { ...m, content: fullContent } : m,
              ),
            );
            break;
          case "done":
            finalSources = event.sources ?? [];
            retrievalMs = event.retrieval_latency_ms ?? retrievalMs;
            break;
          case "error":
            toast.error(event.message || "Stream error");
            break;
        }
      }

      // Finalise the placeholder with metadata
      if (fullContent) {
        const meta: BotMeta = {
          responseTimeMs: retrievalMs,
          generationLatencyMs: 0,
          inputTokens: 0,
          outputTokens: 0,
          sources: finalSources,
          chunks: apiSourcesToChunks(finalSources),
        };
        setMessages((prev) =>
          prev.map((m) =>
            m.id === placeholderId
              ? { ...m, content: fullContent, meta }
              : m,
          ),
        );
        setActiveChunks(meta.chunks);
      }
    } catch (err: any) {
      toast.error(err.message || "Stream failed");
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholderId
            ? {
                ...m,
                content:
                  fullContent ||
                  "⚠️ Streaming failed. Please try again.",
              }
            : m,
        ),
      );
    } finally {
      setStreaming(false);
    }
  };

  // ── Upload ─────────────────────────────────────────────
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const result = await uploadDocument(file);
      toast.success(
        `"${result.filename}" uploaded — ${result.chunks_created} chunks indexed.`,
      );
    } catch (err: any) {
      toast.error(err.message || "Upload failed");
    } finally {
      // Reset so the same file can be re-selected
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // ── Feedback ───────────────────────────────────────────
  const handleFeedback = async (messageId: string, rating: number) => {
    try {
      await submitFeedback(messageId, rating);
      toast.success(rating > 0 ? "Thanks for the feedback!" : "Feedback noted — we'll improve.");
    } catch {
      toast.error("Could not record feedback");
    }
  };

  // ── Render ─────────────────────────────────────────────
  return (
    <div className="flex h-screen w-full flex-col bg-background text-foreground">
      <Toaster position="top-center" />

      {/* Header */}
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-5">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Sparkles className="h-4 w-4" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">RAG Chatbot</div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
              </span>
              Assistant online{streaming && " — streaming…"}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Upload button */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.csv,application/pdf,text/plain,text/csv"
            onChange={handleUpload}
            className="hidden"
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            className="gap-2"
          >
            <Paperclip className="h-4 w-4" />
            <span className="hidden sm:inline">Upload PDF/TXT</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen((v) => !v)}
            className="gap-2"
          >
            {sidebarOpen ? (
              <PanelRightClose className="h-4 w-4" />
            ) : (
              <PanelRightOpen className="h-4 w-4" />
            )}
            <span className="hidden sm:inline">
              {sidebarOpen ? "Hide" : "Show"} debug
            </span>
          </Button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Chat column */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto px-4 py-6 sm:px-8"
          >
            {loadingHistory ? (
              <div className="flex items-center justify-center py-20 text-sm text-muted-foreground">
                Loading conversation…
              </div>
            ) : messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <Sparkles className="mb-4 h-10 w-10 text-muted-foreground" />
                <p className="mb-2 text-lg font-medium">
                  Ask me anything about your documents
                </p>
                <p className="max-w-md text-sm text-muted-foreground">
                  Upload a PDF or TXT file, then ask questions. I'll find the
                  most relevant content and answer from your documents.
                </p>
              </div>
            ) : (
              <div className="mx-auto flex max-w-3xl flex-col gap-6">
                {messages.map((m) => (
                  <MessageBubble
                    key={m.id}
                    message={m}
                    onFeedback={handleFeedback}
                    onInspect={(chunks) => {
                      setActiveChunks(chunks ?? []);
                      setSidebarOpen(true);
                    }}
                  />
                ))}
                {streaming &&
                  messages[messages.length - 1]?.content === "" && (
                    <div className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />
                      Generating…
                    </div>
                  )}
              </div>
            )}
          </div>

          {/* Composer */}
          <div className="border-t border-border bg-card/60 px-4 py-4 backdrop-blur sm:px-8">
            <div className="mx-auto max-w-3xl">
              <div className="flex items-end gap-2 rounded-2xl border border-border bg-background p-2 shadow-sm focus-within:ring-2 focus-within:ring-ring/40">
                <Textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (e.metaKey || e.ctrlKey) {
                        handleSendStream();
                      } else {
                        handleSend();
                      }
                    }
                  }}
                  rows={1}
                  placeholder="Ask about your documents…"
                  disabled={streaming}
                  className="min-h-[40px] resize-none border-0 bg-transparent px-2 py-2 text-sm shadow-none focus-visible:ring-0"
                />
                <Button
                  size="icon"
                  onClick={handleSend}
                  disabled={!input.trim() || streaming}
                  className="h-9 w-9 shrink-0 rounded-xl"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
              <div className="mt-2 flex items-center justify-between px-1 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <Zap className="h-3 w-3" />
                  Mimo 2.5 Pro · RAG pipeline
                </span>
                <span>
                  Enter to send · Ctrl+Enter to stream · Shift+Enter for
                  newline
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="hidden w-[340px] shrink-0 flex-col border-l border-border bg-muted/30 md:flex">
            <div className="flex items-center justify-between border-b border-border bg-card px-4 py-3">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-primary" />
                <h2 className="text-sm font-semibold">
                  RAG Context & Debugging
                </h2>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="rounded-md p-1 text-muted-foreground hover:bg-accent"
                aria-label="Close sidebar"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <div className="mb-3 text-xs uppercase tracking-wide text-muted-foreground">
                Retrieved sources · {activeChunks.length}
              </div>
              <div className="flex flex-col gap-3">
                {activeChunks.map((c, i) => (
                  <ChunkCard key={c.id} chunk={c} rank={i + 1} />
                ))}
                {activeChunks.length === 0 && (
                  <p className="text-sm text-muted-foreground">
                    No retrieval performed for this turn.
                  </p>
                )}
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

// ── Message Bubble ──────────────────────────────────────────────────

function MessageBubble({
  message,
  onFeedback,
  onInspect,
}: {
  message: Message;
  onFeedback: (messageId: string, rating: number) => void;
  onInspect: (chunks: Chunk[] | undefined) => void;
}) {
  const isUser = message.role === "user";
  const meta = message.meta;

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-secondary text-secondary-foreground",
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div
        className={cn("flex max-w-[80%] flex-col", isUser && "items-end")}
      >
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "rounded-tr-sm bg-primary text-primary-foreground"
              : "rounded-tl-sm bg-muted text-foreground",
          )}
        >
          {message.content || (
            <span className="italic text-muted-foreground">Thinking…</span>
          )}
        </div>

        {/* Metadata row (bot messages only) */}
        {!isUser && meta && (
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            {meta.responseTimeMs > 0 && (
              <span>Response: {(meta.responseTimeMs / 1000).toFixed(1)}s</span>
            )}
            {meta.generationLatencyMs > 0 && (
              <span>Gen: {(meta.generationLatencyMs / 1000).toFixed(1)}s</span>
            )}
            {meta.inputTokens > 0 && (
              <span>
                Tokens: {meta.inputTokens}→{meta.outputTokens}
              </span>
            )}
            <div className="flex items-center gap-1">
              <button
                aria-label="Thumbs up"
                onClick={() => onFeedback(message.id, 1.0)}
                className="rounded-md p-1.5 text-muted-foreground transition hover:bg-accent hover:text-foreground"
              >
                <ThumbsUp className="h-3.5 w-3.5" />
              </button>
              <button
                aria-label="Thumbs down"
                onClick={() => onFeedback(message.id, -1.0)}
                className="rounded-md p-1.5 text-muted-foreground transition hover:bg-accent hover:text-foreground"
              >
                <ThumbsDown className="h-3.5 w-3.5" />
              </button>
            </div>
            {meta.sources.length > 0 && (
              <button
                onClick={() => onInspect(meta.chunks)}
                className="ml-auto inline-flex items-center gap-1 text-primary hover:underline"
              >
                <Database className="h-3 w-3" />
                {meta.sources.length} source{meta.sources.length > 1 && "s"}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Chunk Card ──────────────────────────────────────────────────────

function ChunkCard({ chunk, rank }: { chunk: Chunk; rank: number }) {
  const pct = Math.round(chunk.score * 100);
  return (
    <div className="rounded-xl border border-border bg-card p-3 shadow-sm">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-primary/10 text-[10px] font-semibold text-primary">
            {rank}
          </span>
          <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate text-xs font-medium" title={chunk.source}>
            {chunk.source}
          </span>
        </div>
        {pct > 0 && (
          <span
            className={cn(
              "shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold",
              pct >= 85
                ? "bg-emerald-500/10 text-emerald-600"
                : pct >= 70
                  ? "bg-primary/10 text-primary"
                  : "bg-muted text-muted-foreground",
            )}
          >
            {pct}%
          </span>
        )}
      </div>
      {chunk.snippet ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          {chunk.snippet}
        </p>
      ) : (
        <p className="text-xs italic text-muted-foreground">
          Source document — retrieved from vector search.
        </p>
      )}
    </div>
  );
}
