import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  Bot,
  ChevronRight,
  Database,
  FileText,
  PanelRightClose,
  PanelRightOpen,
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

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Aurora Support — AI Customer Assistant" },
      { name: "description", content: "Enterprise AI customer support console with retrieval-augmented context and transparent debugging." },
      { property: "og:title", content: "Aurora Support — AI Customer Assistant" },
      { property: "og:description", content: "Enterprise AI customer support console with retrieval-augmented context and transparent debugging." },
    ],
  }),
  component: Index,
});

type Chunk = {
  id: string;
  source: string;
  score: number;
  snippet: string;
  page?: number;
};

type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  responseTimeSec?: number;
  tokens?: number;
  chunks?: Chunk[];
};

const SEED_MESSAGES: Message[] = [
  {
    id: "m1",
    role: "user",
    content: "Hi! My order #A-10423 hasn't arrived yet. It's been 8 days.",
  },
  {
    id: "m2",
    role: "bot",
    content:
      "Thanks for reaching out. Standard shipping to your region typically takes **5–7 business days**, and orders over 7 days qualify for a free expedited reshipment. I can either open a trace with the carrier or send a replacement today — which would you prefer?",
    responseTimeSec: 1.2,
    tokens: 45,
    chunks: [
      {
        id: "c1",
        source: "shipping-policy-v3.pdf",
        page: 2,
        score: 0.92,
        snippet:
          "Orders not delivered within 7 business days of dispatch are eligible for a free expedited reshipment at the customer's request.",
      },
      {
        id: "c2",
        source: "carrier-sla.md",
        score: 0.81,
        snippet:
          "Standard ground service: 5–7 business days within the continental zone. Trace requests open within 24h.",
      },
      {
        id: "c3",
        source: "returns-handbook.pdf",
        page: 11,
        score: 0.64,
        snippet:
          "Replacement orders bypass the standard fulfillment queue and ship priority on the same business day.",
      },
    ],
  },
];

const FALLBACK_CHUNKS: Chunk[] = [
  {
    id: "f1",
    source: "faq-general.md",
    score: 0.74,
    snippet:
      "Customer support responses should cite policy when available and offer a concrete next action.",
  },
  {
    id: "f2",
    source: "tone-guidelines.pdf",
    page: 4,
    score: 0.58,
    snippet: "Maintain a warm, concise tone. Avoid hedging language.",
  },
];

function Index() {
  const [messages, setMessages] = useState<Message[]>(SEED_MESSAGES);
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeChunks, setActiveChunks] = useState<Chunk[]>(
    SEED_MESSAGES[1].chunks ?? [],
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    const botMsg: Message = {
      id: crypto.randomUUID(),
      role: "bot",
      content:
        "Got it — I've pulled the most relevant policies. Based on what I found, here's the recommended next step. Let me know if you'd like me to take action on your behalf.",
      responseTimeSec: Number((0.8 + Math.random() * 1.4).toFixed(1)),
      tokens: Math.floor(30 + Math.random() * 80),
      chunks: FALLBACK_CHUNKS,
    };
    setMessages((m) => [...m, userMsg, botMsg]);
    setActiveChunks(FALLBACK_CHUNKS);
    setInput("");
  };

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
            <div className="text-sm font-semibold">Aurora Outfitters</div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
              </span>
              Assistant online
            </div>
          </div>
        </div>
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
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Chat column */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto px-4 py-6 sm:px-8"
          >
            <div className="mx-auto flex max-w-3xl flex-col gap-6">
              {messages.map((m) => (
                <MessageBubble
                  key={m.id}
                  message={m}
                  onInspect={(chunks) => {
                    setActiveChunks(chunks);
                    setSidebarOpen(true);
                  }}
                />
              ))}
            </div>
          </div>

          {/* Composer */}
          <div className="border-t border-border bg-card/60 backdrop-blur px-4 py-4 sm:px-8">
            <div className="mx-auto max-w-3xl">
              <div className="flex items-end gap-2 rounded-2xl border border-border bg-background p-2 shadow-sm focus-within:ring-2 focus-within:ring-ring/40">
                <Textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  rows={1}
                  placeholder="Ask about an order, return, or product…"
                  className="min-h-[40px] resize-none border-0 bg-transparent px-2 py-2 text-sm shadow-none focus-visible:ring-0"
                />
                <Button
                  size="icon"
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="h-9 w-9 shrink-0 rounded-xl"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
              <div className="mt-2 flex items-center justify-between px-1 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <Zap className="h-3 w-3" />
                  Mimo 2.5 Pro
                </span>
                <span>Press Enter to send · Shift+Enter for newline</span>
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
                <h2 className="text-sm font-semibold">RAG Context & Debugging</h2>
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
                Retrieved chunks · {activeChunks.length}
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

function MessageBubble({
  message,
  onInspect,
}: {
  message: Message;
  onInspect: (chunks: Chunk[]) => void;
}) {
  const isUser = message.role === "user";
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
      <div className={cn("flex max-w-[80%] flex-col", isUser && "items-end")}>
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "rounded-tr-sm bg-primary text-primary-foreground"
              : "rounded-tl-sm bg-muted text-foreground",
          )}
        >
          {message.content}
        </div>

        {!isUser && (
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            {message.responseTimeSec !== undefined && (
              <span>Response Time: {message.responseTimeSec}s</span>
            )}
            {message.tokens !== undefined && (
              <span>Token Usage: {message.tokens}</span>
            )}
            <div className="flex items-center gap-1">
              <FeedbackButton
                label="Like"
                onClick={() => toast.success("Feedback recorded")}
              >
                <ThumbsUp className="h-3.5 w-3.5" />
              </FeedbackButton>
              <FeedbackButton
                label="Dislike"
                onClick={() => toast("Feedback recorded")}
              >
                <ThumbsDown className="h-3.5 w-3.5" />
              </FeedbackButton>
            </div>
            {message.chunks && message.chunks.length > 0 && (
              <button
                onClick={() => onInspect(message.chunks!)}
                className="ml-auto inline-flex items-center gap-1 text-primary hover:underline"
              >
                <Database className="h-3 w-3" />
                {message.chunks.length} sources
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function FeedbackButton({
  children,
  label,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-label={label}
      onClick={onClick}
      className="rounded-md p-1.5 text-muted-foreground transition hover:bg-accent hover:text-foreground"
    >
      {children}
    </button>
  );
}

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
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground">
        {chunk.snippet}
      </p>
      {chunk.page !== undefined && (
        <div className="mt-2 text-[10px] uppercase tracking-wide text-muted-foreground">
          Page {chunk.page}
        </div>
      )}
    </div>
  );
}
