/**
 * API client for the RAG Chatbot backend.
 *
 * All calls go to the FastAPI backend running on localhost:8000.
 * Each request generates or reuses a session_id so conversations persist.
 */

const API_BASE = "http://localhost:8000";

// ── Session management ──────────────────────────────────────────────

function getSessionId(): string {
  if (typeof window === "undefined") return "";
  let id = sessionStorage.getItem("rag_session_id");
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem("rag_session_id", id);
  }
  return id;
}

// ── Types ───────────────────────────────────────────────────────────

export interface ChatPayload {
  session_id: string;
  message: string;
}

export interface ChatResponse {
  id: string;
  session_id: string;
  message: string;
  sources: string[];
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  fallback_used: boolean;
  timestamp: string;
}

export interface UploadResponse {
  filename: string;
  chunks_created: number;
  duplicates_skipped: number;
  message: string;
}

export interface HistoryMessage {
  id: string;
  role: string;
  message: string;
  timestamp: string;
}

export interface HistoryResponse {
  session_id: string;
  messages: HistoryMessage[];
}

export interface StreamEvent {
  type: "status" | "token" | "done" | "error";
  message?: string;
  token?: string;
  chunks_found?: number;
  sources?: string[];
  retrieval_latency_ms?: number;
  fallback_used?: boolean;
  error?: string;
}

// ── Non-streaming chat ──────────────────────────────────────────────

export async function sendMessage(message: string): Promise<ChatResponse> {
  const payload: ChatPayload = {
    session_id: getSessionId(),
    message,
  };

  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Chat request failed");
  }

  return res.json();
}

// ── Streaming chat (SSE) ────────────────────────────────────────────

export async function* streamMessage(
  message: string,
): AsyncGenerator<StreamEvent> {
  const payload: ChatPayload = {
    session_id: getSessionId(),
    message,
  };

  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Stream request failed");
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    // Keep the last (possibly incomplete) line in the buffer
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const json: StreamEvent = JSON.parse(line.slice(6));
        yield json;
      } catch {
        // Skip malformed lines
      }
    }
  }
}

// ── Chat history ────────────────────────────────────────────────────

export async function getHistory(
  sessionId?: string,
): Promise<HistoryResponse> {
  const id = sessionId ?? getSessionId();
  const res = await fetch(`${API_BASE}/chat/${id}`);

  if (!res.ok) {
    if (res.status === 404) {
      return { session_id: id, messages: [] };
    }
    throw new Error("Failed to load history");
  }

  return res.json();
}

// ── Feedback ────────────────────────────────────────────────────────

export async function submitFeedback(
  messageId: string,
  rating: number, // 1.0 = thumbs up, -1.0 = thumbs down
  comment?: string,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message_id: messageId,
      rating,
      comment: comment ?? null,
    }),
  });

  if (!res.ok) throw new Error("Failed to submit feedback");
}

// ── Document management ─────────────────────────────────────────────

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

export interface DocInfo {
  filename: string;
  chunks: number;
  uploaded_at: string;
}

export async function listDocuments(): Promise<DocInfo[]> {
  const res = await fetch(`${API_BASE}/upload`);
  if (!res.ok) throw new Error("Failed to list documents");
  return res.json();
}

export interface DeleteResponse {
  filename: string;
  chunks_deleted: number;
  message: string;
}

export async function deleteDocument(
  filename: string,
): Promise<DeleteResponse> {
  const res = await fetch(
    `${API_BASE}/upload/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  );

  if (!res.ok) {
    if (res.status === 404) {
      throw new Error(`No document named "${filename}" found.`);
    }
    throw new Error("Failed to delete document");
  }

  return res.json();
}
