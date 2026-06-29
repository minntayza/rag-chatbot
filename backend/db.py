"""
Database client using Supabase Python SDK.

Connects via REST API (HTTPS) — no direct PostgreSQL connection needed.
This avoids IPv6/Supavisor issues and works from any network.

pgvector operations use the `match_documents` database function
created in Supabase SQL Editor.
"""

from __future__ import annotations

from supabase import create_client, Client

from config import get_settings

settings = get_settings()

# ── Supabase client (singleton) ──────────────────────────────────────
_supabase_client: Client | None = None


def get_supabase() -> Client:
    """
    Return a cached Supabase client.

    Uses the service role key for server-side operations (bypasses RLS).
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )
    return _supabase_client


async def init_db() -> None:
    """
    Verify the database connection by running a simple query.
    Tables should already exist (created via SQL Editor).
    """
    client = get_supabase()
    # Test connection by querying chat_history
    client.table("chat_history").select("id").limit(1).execute()


async def close_db() -> None:
    """No-op — the Supabase client doesn't need explicit cleanup."""
    pass
