"""Multi-provider LLM pool with round-robin + cross-tier failover.

Problem this solves
-------------------
Free-tier Gemini is 5 rpm / 20 requests per day per key. A single-key setup
falls over halfway through the Day-4 eval and blocks the demo. This pool
lets one deployment hold:

* 2+ Gemini keys (round-robin across them → 10 rpm, 40 requests/day)
* 2+ HuggingFace keys (round-robin across them)
* Optional Anthropic key
* Automatic failover: if the whole Gemini tier is in cooldown, calls
  transparently switch to the HuggingFace tier — the caller never has to
  know that a quota tripped.

Role-based routing
------------------
`.generate(...)` accepts a `tier_order` override so different call sites can
prefer different backends:

* Answer generation → default order (`gemini,huggingface,anthropic`) — judges
  see the sharper Gemini output.
* Claim verification → override to `huggingface,gemini` — verification is a
  one-token classifier, Qwen 7B handles it fine, and this preserves Gemini
  quota for actual answers.

Config
------
All optional. The pool auto-discovers whatever exists:

  GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ...
  HF_TOKEN, HF_TOKEN_2, HF_TOKEN_3, ...
  ANTHROPIC_API_KEY

  KIDNEY_RAG_GEMINI_MODEL    (default: gemini-3.6-flash)
  KIDNEY_RAG_HF_MODEL        (default: Qwen/Qwen2.5-7B-Instruct)
  KIDNEY_RAG_ANTHROPIC_MODEL (default: claude-sonnet-5)

  LLM_POOL_ORDER             (default: gemini,huggingface,anthropic)

Cooldown
--------
On a 429 / RESOURCE_EXHAUSTED / quota error we parse the provider's suggested
retry-after (Gemini emits `retryDelay: Ns`; other providers may say
`retry in Ns`) and mark that provider unavailable until it clears. Any other
error type bubbles up unchanged — the pool won't paper over a real bug.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("kidney_rag.llm_pool")

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

DEFAULT_TIER_ORDER = ["gemini", "huggingface", "anthropic"]
# Verification prefers HF (cheap, throwaway binary classification) so Gemini
# quota is preserved for judge-facing answers.
DEFAULT_VERIFY_TIER_ORDER = ["huggingface", "gemini", "anthropic"]

DEFAULT_COOLDOWN_S = 60.0    # fallback when the provider doesn't hint one


class LLMPoolExhausted(RuntimeError):
    """All providers in the requested tier order are unavailable."""


@dataclass
class Provider:
    backend: str            # "gemini" | "huggingface" | "anthropic"
    key_env: str            # env var name — used as public identifier; the
                            # api_key itself is repr=False so it doesn't leak
                            # into logs / /health responses.
    api_key: str = field(repr=False)
    model: str
    tier: str               # usually == backend, kept separate to allow
                            # future custom groupings (e.g. gemini-pro tier).

    _client: Any = field(default=None, repr=False)
    cooldown_until: float = 0.0
    last_error: str | None = None
    successes: int = 0
    failures: int = 0

    @property
    def id(self) -> str:
        return f"{self.backend}:{self.key_env}"

    def in_cooldown(self, now: float | None = None) -> bool:
        return self.cooldown_until > (now if now is not None else time.time())

    def cooldown_remaining_s(self, now: float | None = None) -> float:
        return max(0.0, self.cooldown_until - (now if now is not None else time.time()))


class LLMPool:
    """Ordered tier pool with RR within tier + failover across tiers."""

    def __init__(self, tier_order: list[str], providers: list[Provider]):
        self.tier_order = list(tier_order)
        self._by_tier: dict[str, list[Provider]] = {t: [] for t in self.tier_order}
        for p in providers:
            self._by_tier.setdefault(p.tier, []).append(p)
        self._rr_idx: dict[str, int] = {t: 0 for t in self._by_tier}

    # --- construction ------------------------------------------------------

    @classmethod
    def from_env(cls, tier_order: list[str] | None = None) -> "LLMPool":
        providers: list[Provider] = []
        for env_name, api_key in _scan_keys("GOOGLE_API_KEY").items():
            providers.append(Provider(
                backend="gemini", key_env=env_name, api_key=api_key,
                model=os.environ.get("KIDNEY_RAG_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
                tier="gemini",
            ))
        for env_name, api_key in _scan_keys("HF_TOKEN").items():
            providers.append(Provider(
                backend="huggingface", key_env=env_name, api_key=api_key,
                model=os.environ.get("KIDNEY_RAG_HF_MODEL", DEFAULT_HF_MODEL),
                tier="huggingface",
            ))
        for env_name, api_key in _scan_keys("ANTHROPIC_API_KEY").items():
            providers.append(Provider(
                backend="anthropic", key_env=env_name, api_key=api_key,
                model=os.environ.get("KIDNEY_RAG_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
                tier="anthropic",
            ))
        order = tier_order or _parse_tier_order(
            os.environ.get("LLM_POOL_ORDER", ",".join(DEFAULT_TIER_ORDER))
        )
        # Drop tiers with no providers so status/output isn't noisy.
        order = [t for t in order if any(p.tier == t for p in providers)]
        pool = cls(order, providers)
        log.info("LLMPool: tiers=%s providers=%s",
                 order, [p.id for p in providers])
        return pool

    # --- introspection -----------------------------------------------------

    def providers(self) -> list[Provider]:
        return [p for t in self.tier_order for p in self._by_tier.get(t, [])]

    def has_providers(self) -> bool:
        return any(self._by_tier.get(t) for t in self.tier_order)

    def status(self) -> list[dict[str, Any]]:
        now = time.time()
        rows = []
        for p in self.providers():
            rows.append({
                "id": p.id,
                "backend": p.backend,
                "key_env": p.key_env,
                "model": p.model,
                "tier": p.tier,
                "status": "cooldown" if p.in_cooldown(now) else "ready",
                "cooldown_remaining_s": round(p.cooldown_remaining_s(now), 1),
                "successes": p.successes,
                "failures": p.failures,
                "last_error": p.last_error,
            })
        return rows

    # --- main entry point --------------------------------------------------

    def generate(self, system_prompt: str, user_message: str,
                 max_tokens: int = 2000,
                 exclude_provider_ids: set[str] | None = None,
                 tier_order: list[str] | None = None) -> tuple[str, Provider]:
        """Try providers in `tier_order` (defaults to pool's), RR within a tier.

        Returns (text, provider_used). Raises LLMPoolExhausted only when
        every tier in the order is entirely in cooldown. Non-transient errors
        (auth failure, bad request, network) bubble up unchanged so callers
        see real bugs instead of a generic "exhausted" message.
        """
        exclude = exclude_provider_ids or set()
        order = tier_order or self.tier_order
        attempts: list[tuple[str, str]] = []

        for tier in order:
            tier_providers = self._by_tier.get(tier, [])
            if not tier_providers:
                continue
            n = len(tier_providers)
            start = self._rr_idx.get(tier, 0)
            for offset in range(n):
                p = tier_providers[(start + offset) % n]
                if p.id in exclude:
                    continue
                if p.in_cooldown():
                    continue
                try:
                    text = self._call(p, system_prompt, user_message, max_tokens)
                except Exception as exc:                        # noqa: BLE001
                    cooldown = self._cooldown_from_error(exc)
                    if cooldown is None:
                        raise           # not rate-limit shaped — real error
                    p.cooldown_until = time.time() + cooldown
                    p.failures += 1
                    p.last_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                    attempts.append((p.id, f"cooldown {cooldown:.0f}s"))
                    log.warning("Provider %s cooling down %.0fs: %s",
                                p.id, cooldown, str(exc)[:200])
                    continue
                # Success — advance RR pointer past this provider.
                self._rr_idx[tier] = (start + offset + 1) % n
                p.successes += 1
                p.last_error = None
                return text, p

        raise LLMPoolExhausted(
            f"All providers exhausted. Order tried: {order}. Attempts: {attempts}"
        )

    # --- backend adapters --------------------------------------------------

    def _client_for(self, p: Provider):
        if p._client is not None:
            return p._client
        if p.backend == "gemini":
            from google import genai
            p._client = genai.Client(api_key=p.api_key)
        elif p.backend == "anthropic":
            from anthropic import Anthropic
            p._client = Anthropic(api_key=p.api_key)
        elif p.backend == "huggingface":
            from huggingface_hub import InferenceClient
            p._client = InferenceClient(model=p.model, token=p.api_key)
        else:
            raise ValueError(f"Unknown backend: {p.backend!r}")
        return p._client

    def _call(self, p: Provider, system_prompt: str, user_message: str,
              max_tokens: int) -> str:
        client = self._client_for(p)
        if p.backend == "gemini":
            from google.genai import types
            response = client.models.generate_content(
                model=p.model, contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text or ""
        if p.backend == "anthropic":
            response = client.messages.create(
                model=p.model, max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return "".join(b.text for b in response.content if b.type == "text")
        # huggingface chat.completions
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    # --- error → cooldown --------------------------------------------------

    # Matches Gemini "retryDelay: 58s", generic "retry in 30s",
    # and Anthropic "retry-after: 12s".
    _RETRY_S_RE = re.compile(
        r"(?:retry[-_]?delay|retry[-_]?after|retry in)\D{0,10}(\d+(?:\.\d+)?)\s*s?",
        re.IGNORECASE,
    )

    @classmethod
    def _cooldown_from_error(cls, exc: Exception) -> float | None:
        """Return seconds to cool this provider down.

        None means "this isn't a rate-limit — bubble it up". A number means
        "hold this provider until now+N, try another provider instead".
        """
        msg = str(exc)
        low = msg.lower()
        transient = (
            "429" in msg
            or "rate limit" in low or "rate_limit" in low or "ratelimit" in low
            or "resource_exhausted" in low or "resourceexhausted" in low
            or "quota" in low
            or "too many requests" in low
            or "overloaded" in low
        )
        if not transient:
            return None
        m = cls._RETRY_S_RE.search(msg)
        if m:
            try:
                return float(m.group(1)) + 1.0        # small safety margin
            except ValueError:
                pass
        return DEFAULT_COOLDOWN_S


# --- helpers ---------------------------------------------------------------

def _scan_keys(prefix: str) -> dict[str, str]:
    """Return {env_name: value} for PREFIX, PREFIX_2, PREFIX_3, ... in order.

    Stops on the first missing suffix (so `HF_TOKEN`, skip, `HF_TOKEN_3` is
    read as just `HF_TOKEN`). This matches how people actually number keys.
    """
    found: dict[str, str] = {}
    v = os.environ.get(prefix)
    if v:
        found[prefix] = v
    i = 2
    while True:
        name = f"{prefix}_{i}"
        v = os.environ.get(name)
        if not v:
            break
        found[name] = v
        i += 1
    return found


def _parse_tier_order(s: str) -> list[str]:
    return [t.strip().lower() for t in s.split(",") if t.strip()]
