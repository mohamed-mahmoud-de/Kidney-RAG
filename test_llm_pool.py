"""Offline tests for llm_pool.py.

No real API calls. Stubbing works by monkey-patching LLMPool._call — the
class-level method that talks to the SDK — so we can drive the round-robin,
cooldown, and failover logic deterministically.
"""
import time

from llm_pool import (
    LLMPool, Provider, LLMPoolExhausted,
    _scan_keys, _parse_tier_order,
    DEFAULT_COOLDOWN_S,
)


def _mk_provider(backend: str, tier: str, key_env: str) -> Provider:
    return Provider(backend=backend, key_env=key_env, api_key="k",
                    model="stub-model", tier=tier)


def _mk_pool(*providers, tier_order=None) -> LLMPool:
    order = tier_order or sorted({p.tier for p in providers})
    return LLMPool(order, list(providers))


# --- Round-robin within a tier -------------------------------------------

def test_round_robin_within_tier(monkeypatch):
    """Two Gemini keys should be hit alternately across calls."""
    a = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    b = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY_2")
    pool = _mk_pool(a, b, tier_order=["gemini"])

    used: list[str] = []
    monkeypatch.setattr(LLMPool, "_call",
                        lambda self, p, s, u, m: (used.append(p.id), "ok")[1])

    for _ in range(4):
        text, p = pool.generate("sys", "usr")
        assert text == "ok"

    # A, B, A, B  — pure alternation
    assert used == [a.id, b.id, a.id, b.id]
    assert a.successes == 2 and b.successes == 2


def test_cooldown_skips_provider(monkeypatch):
    """A provider whose cooldown hasn't expired is skipped, not tried."""
    a = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    b = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY_2")
    pool = _mk_pool(a, b, tier_order=["gemini"])
    a.cooldown_until = time.time() + 300     # 5 min in the future

    used: list[str] = []
    monkeypatch.setattr(LLMPool, "_call",
                        lambda self, p, s, u, m: (used.append(p.id), "ok")[1])

    for _ in range(3):
        pool.generate("sys", "usr")

    # All calls go to b; a stays cooling
    assert used == [b.id, b.id, b.id]


def test_cooldown_expires(monkeypatch):
    """Once cooldown_until is in the past, provider re-enters rotation."""
    a = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    pool = _mk_pool(a, tier_order=["gemini"])
    a.cooldown_until = time.time() - 1        # already expired

    monkeypatch.setattr(LLMPool, "_call", lambda self, p, s, u, m: "ok")
    text, _ = pool.generate("sys", "usr")
    assert text == "ok"
    assert a.successes == 1


# --- Cross-tier failover -------------------------------------------------

def test_failover_across_tiers(monkeypatch):
    """When the whole primary tier is in cooldown, calls fall to the next tier."""
    g = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    h = _mk_provider("huggingface", "huggingface", "HF_TOKEN")
    pool = _mk_pool(g, h, tier_order=["gemini", "huggingface"])
    g.cooldown_until = time.time() + 300

    used: list[str] = []
    monkeypatch.setattr(LLMPool, "_call",
                        lambda self, p, s, u, m: (used.append(p.id), "ok")[1])

    pool.generate("sys", "usr")
    assert used == [h.id]


def test_transient_error_triggers_cooldown_and_fails_over(monkeypatch):
    """A 429 on Gemini cools Gemini down and the request retries on HF."""
    g = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    h = _mk_provider("huggingface", "huggingface", "HF_TOKEN")
    pool = _mk_pool(g, h, tier_order=["gemini", "huggingface"])

    call_log: list[str] = []

    def fake_call(self, p, s, u, m):
        call_log.append(p.id)
        if p.backend == "gemini":
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 42s.")
        return "hf-answer"

    monkeypatch.setattr(LLMPool, "_call", fake_call)

    text, used = pool.generate("sys", "usr")
    assert text == "hf-answer"
    assert used.id == h.id
    # Gemini was tried, cooled down; then HF was tried and won.
    assert call_log == [g.id, h.id]
    assert g.in_cooldown()
    # Retry-after was parsed (42s + 1s safety = ~43s)
    assert 40 <= g.cooldown_remaining_s() <= 45
    assert g.failures == 1 and h.successes == 1


def test_non_transient_error_bubbles_up(monkeypatch):
    """A real bug (bad request, auth error) is NOT caught — must surface."""
    g = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    pool = _mk_pool(g, tier_order=["gemini"])

    def fake_call(self, p, s, u, m):
        raise ValueError("400 Bad Request: prompt too long")

    monkeypatch.setattr(LLMPool, "_call", fake_call)

    import pytest
    with pytest.raises(ValueError, match="Bad Request"):
        pool.generate("sys", "usr")
    assert not g.in_cooldown()
    assert g.failures == 0                   # not counted as a rate-limit failure


def test_exhaustion_raises_llm_pool_exhausted(monkeypatch):
    """All providers in cooldown → LLMPoolExhausted."""
    g = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    h = _mk_provider("huggingface", "huggingface", "HF_TOKEN")
    pool = _mk_pool(g, h, tier_order=["gemini", "huggingface"])
    g.cooldown_until = h.cooldown_until = time.time() + 300

    monkeypatch.setattr(LLMPool, "_call",
                        lambda self, p, s, u, m: (_ for _ in ()).throw(AssertionError("not called")))

    import pytest
    with pytest.raises(LLMPoolExhausted):
        pool.generate("sys", "usr")


# --- Role-based tier override --------------------------------------------

def test_tier_order_override_routes_to_secondary(monkeypatch):
    """Passing tier_order lets one call prefer a non-default tier
    (verification calls sending traffic to HF while generation stays on Gemini)."""
    g = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    h = _mk_provider("huggingface", "huggingface", "HF_TOKEN")
    pool = _mk_pool(g, h, tier_order=["gemini", "huggingface"])

    used: list[str] = []
    monkeypatch.setattr(LLMPool, "_call",
                        lambda self, p, s, u, m: (used.append(p.id), "ok")[1])

    pool.generate("sys", "usr", tier_order=["huggingface", "gemini"])
    assert used == [h.id]                    # HF picked first because of override
    assert g.successes == 0


def test_exclude_provider_skips_it(monkeypatch):
    """A same-request follow-up can avoid the provider it just used."""
    g1 = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    g2 = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY_2")
    pool = _mk_pool(g1, g2, tier_order=["gemini"])

    used: list[str] = []
    monkeypatch.setattr(LLMPool, "_call",
                        lambda self, p, s, u, m: (used.append(p.id), "ok")[1])

    pool.generate("sys", "usr", exclude_provider_ids={g1.id})
    assert used == [g2.id]


# --- Cooldown parsing ----------------------------------------------------

def test_retry_after_parsed_from_gemini_style_error():
    exc = RuntimeError("429 RESOURCE_EXHAUSTED. retryDelay: 58s")
    assert LLMPool._cooldown_from_error(exc) == 59.0     # 58 + 1s margin


def test_retry_after_parsed_from_generic_style_error():
    exc = RuntimeError("429 Please retry in 12.5s")
    assert LLMPool._cooldown_from_error(exc) == 13.5


def test_no_retry_after_uses_default_cooldown():
    exc = RuntimeError("429 Too Many Requests")
    assert LLMPool._cooldown_from_error(exc) == DEFAULT_COOLDOWN_S


def test_non_rate_limit_error_returns_none():
    assert LLMPool._cooldown_from_error(ValueError("bad prompt")) is None
    assert LLMPool._cooldown_from_error(RuntimeError("500 server error")) is None


# --- Env scanning --------------------------------------------------------

def test_scan_keys_reads_numbered_suffixes(monkeypatch):
    monkeypatch.setenv("MY_KEY", "a")
    monkeypatch.setenv("MY_KEY_2", "b")
    monkeypatch.setenv("MY_KEY_3", "c")
    monkeypatch.delenv("MY_KEY_4", raising=False)
    monkeypatch.setenv("MY_KEY_5", "e")               # ignored — gap at _4

    found = _scan_keys("MY_KEY")
    assert found == {"MY_KEY": "a", "MY_KEY_2": "b", "MY_KEY_3": "c"}


def test_scan_keys_returns_empty_when_missing(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    assert _scan_keys("NOPE") == {}


def test_parse_tier_order_normalizes():
    assert _parse_tier_order("Gemini, HuggingFace , anthropic") == [
        "gemini", "huggingface", "anthropic",
    ]
    assert _parse_tier_order("") == []


# --- Introspection -------------------------------------------------------

def test_status_marks_cooldown_and_ready():
    g = _mk_provider("gemini", "gemini", "GOOGLE_API_KEY")
    h = _mk_provider("huggingface", "huggingface", "HF_TOKEN")
    pool = _mk_pool(g, h, tier_order=["gemini", "huggingface"])
    g.cooldown_until = time.time() + 30

    status = pool.status()
    ids = {row["id"]: row for row in status}
    assert ids[g.id]["status"] == "cooldown"
    assert 25 <= ids[g.id]["cooldown_remaining_s"] <= 31
    assert ids[h.id]["status"] == "ready"


def test_env_discovery_picks_up_groq_keys(monkeypatch):
    """LLMPool.from_env() must include GROQ_API_KEY[_N] as a discovered tier."""
    # Isolate — clear ALL known key env vars so from_env() sees only ours.
    for var in ("GOOGLE_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
        for i in range(2, 5):
            monkeypatch.delenv(f"{var}_{i}", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_first")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk_second")
    pool = LLMPool.from_env()
    ids = [p.id for p in pool.providers()]
    assert ids == ["groq:GROQ_API_KEY", "groq:GROQ_API_KEY_2"]
    # Groq should be a discovered tier — order includes "groq" first per
    # the default tier order.
    assert "groq" in pool.tier_order


def test_groq_call_maps_success(monkeypatch):
    """Stub Groq's HTTP session to return a valid OpenAI-shaped response."""
    from llm_pool import LLMPool, Provider

    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "OK from Groq"}}]}

    class FakeSession:
        def __init__(self):
            self.headers = {}
        def post(self, url, json=None, timeout=None):
            assert "chat/completions" in url
            assert json["model"] == "llama-3.1-8b-instant"
            return FakeResp()

    p = Provider(backend="groq", key_env="GROQ_API_KEY",
                 api_key="gsk_x", model="llama-3.1-8b-instant", tier="groq")
    p._client = FakeSession()
    pool = LLMPool(["groq"], [p])
    text, provider = pool.generate("sys", "usr")
    assert text == "OK from Groq"
    assert provider.id == "groq:GROQ_API_KEY"
    assert provider.successes == 1


def test_groq_429_triggers_cooldown_and_failover(monkeypatch):
    """A Groq 429 must be classified as transient and drop to the next tier."""
    from llm_pool import LLMPool, Provider

    class FakeResp:
        status_code = 429
        text = "429 rate_limit_exceeded. retry-after: 30s"

    class FailingSession:
        def __init__(self):
            self.headers = {}
        def post(self, *_a, **_k):
            return FakeResp()

    g = Provider(backend="groq", key_env="GROQ_API_KEY",
                 api_key="gsk_x", model="llama-3.1-8b-instant", tier="groq")
    g._client = FailingSession()
    h = _mk_provider("huggingface", "huggingface", "HF_TOKEN")
    pool = LLMPool(["groq", "huggingface"], [g, h])

    # Grab the real _call once, then patch it to route HF calls to a stub and
    # let the Groq branch fall through to the real code (which will hit the
    # FailingSession and raise the 429-shaped RuntimeError we want).
    real_call = LLMPool._call
    def dispatch(self, p, s, u, m):
        if p.backend == "huggingface":
            return "hf-fallback"
        return real_call(self, p, s, u, m)
    monkeypatch.setattr(LLMPool, "_call", dispatch)

    text, used = pool.generate("sys", "usr")
    assert text == "hf-fallback"
    assert used.id == h.id
    assert g.in_cooldown()
    # Retry-after 30s was parsed by _cooldown_from_error → ~31s left.
    assert 28 <= g.cooldown_remaining_s() <= 32


def test_status_does_not_leak_api_keys():
    g = Provider(backend="gemini", key_env="GOOGLE_API_KEY",
                 api_key="SECRET_XYZ", model="m", tier="gemini")
    pool = LLMPool(["gemini"], [g])
    status = pool.status()
    text = str(status)
    assert "SECRET_XYZ" not in text                # api_key is never surfaced
    assert "GOOGLE_API_KEY" in text                # env name is fine


if __name__ == "__main__":
    import sys
    try:
        import pytest
    except ImportError:
        print("pytest not installed — install with: pip install pytest")
        sys.exit(2)
    sys.exit(pytest.main([__file__, "-v"]))
