#!/usr/bin/env python3
"""Comprehensive Stress, Edge-Case, and Concurrency Test Suite for kusi-qwen-api (Ornith-1.5-35B-Uncensored).

Verifies all 20 critical, medium, and minor items from Kwaku's Stress Test Report against
https://kusi-qwen-api.vercel.app/v1.
"""

import concurrent.futures
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://kusi-qwen-api.vercel.app"
API_KEY = (pathlib.Path.home() / ".config" / "ornith" / "api_key").read_text().strip()


def http_req(method: str, path: str, body=None, key: str = API_KEY, raw_body: bytes = None, timeout: float = 180.0):
    url = BASE_URL + path
    headers = {"User-Agent": "Ornith-Stress-Verifier/2.0"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if raw_body is not None:
        data = raw_body
        headers["Content-Type"] = "application/json"
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            return r.status, hdrs, parsed, time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        hdrs = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw
        return e.code, hdrs, parsed, time.time() - t0


def run_suite():
    results = []

    def check(name: str, fn):
        t0 = time.time()
        try:
            detail = fn()
            dt = time.time() - t0
            print(f"  [PASS] {name} ({dt:.2f}s) -> {detail}")
            results.append((name, True, detail))
        except Exception as e:
            dt = time.time() - t0
            print(f"  [FAIL] {name} ({dt:.2f}s) -> {e}")
            results.append((name, False, str(e)))

    print(f"=== Running Ornith API Stress & Edge-Case Suite against {BASE_URL} ===\n")

    # 1. Health & Availability Check
    def test_health():
        st, _, data, _ = http_req("GET", "/health", key=None)
        assert st == 200 and data.get("ok") is True, f"Expected 200 ok=True, got {st} {data}"
        return f"status={st}, model={data.get('model')}"
    check("1. GET /health endpoint", test_health)

    # 2. Missing & Bad API Key -> 401 JSON
    def test_auth():
        st1, _, d1, _ = http_req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, key=None)
        st2, _, d2, _ = http_req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, key="sk-bad-key")
        assert st1 == 401 and isinstance(d1, dict) and "error" in d1, f"Missing key got {st1} {d1}"
        assert st2 == 401 and isinstance(d2, dict) and "error" in d2, f"Bad key got {st2} {d2}"
        return f"missing={st1}, bad={st2}"
    check("2. Auth rejection (401 JSON)", test_auth)

    # 3. Wrong HTTP Method -> 405 Method Not Allowed (NOT 404)
    def test_wrong_method():
        st, hdrs, d, _ = http_req("GET", "/v1/chat/completions")
        assert st == 405, f"Expected 405, got {st} {d}"
        assert isinstance(d, dict) and "error" in d, f"Expected JSON error, got {d}"
        return f"status={st}, allow={hdrs.get('allow')}, code={d['error'].get('code')}"
    check("3. GET /v1/chat/completions returns 405 Method Not Allowed", test_wrong_method)

    # 4. Malformed JSON -> 400 & Unknown Path -> 404
    def test_malformed_and_404():
        st1, _, d1, _ = http_req("POST", "/v1/chat/completions", raw_body=b"{not valid json")
        st2, _, d2, _ = http_req("GET", "/v1/does-not-exist")
        assert st1 == 400 and isinstance(d1, dict) and "error" in d1, f"Malformed JSON got {st1} {d1}"
        assert st2 == 404 and isinstance(d2, dict) and "error" in d2, f"Unknown path got {st2} {d2}"
        return f"malformed={st1}, unknown_path={st2}"
    check("4. Malformed JSON (400) & Unknown Path (404)", test_malformed_and_404)

    # 5. GET /v1/models deduplicated
    def test_models_dedup():
        st, _, d, _ = http_req("GET", "/v1/models")
        assert st == 200 and isinstance(d, dict), f"Got {st} {d}"
        ids = [m["id"] for m in d.get("data", [])]
        assert len(ids) == len(set(ids)), f"Duplicate model IDs found: {ids}"
        assert "ornith-1.5-35b-uncensored" in ids and "ornith-calendar" in ids, f"Missing models: {ids}"
        return f"models={ids} (0 duplicates)"
    check("5. GET /v1/models deduplicated", test_models_dedup)

    # 6. Unknown model name -> 404 model_not_found
    def test_unknown_model():
        st, _, d, _ = http_req("POST", "/v1/chat/completions", {
            "model": "does-not-exist-9000",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert st == 404, f"Expected 404, got {st} {d}"
        assert d["error"]["code"] == "model_not_found", f"Expected model_not_found, got {d}"
        return f"status={st}, code={d['error']['code']}"
    check("6. Unknown model returns 404 model_not_found", test_unknown_model)

    # 7. Omitted model field works & uses default
    def test_omitted_model():
        st, _, d, _ = http_req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Reply with one word: ok"}],
            "max_tokens": 8,
        })
        assert st == 200 and d.get("model") == "ornith-1.5-35b-uncensored", f"Got {st} {d}"
        return f"status={st}, model={d.get('model')}"
    check("7. Omitted model defaults cleanly", test_omitted_model)

    # 8. Distinct model identities & low prompt token count (< 35 tokens instead of 1,045!)
    def test_models_and_tokens():
        st1, _, d1, _ = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        })
        st2, _, d2, _ = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-calendar",
            "messages": [{"role": "user", "content": "What model are you? Reply in one short sentence."}],
            "max_tokens": 30,
        })
        assert st1 == 200 and st2 == 200, f"Got {st1}, {st2}"
        pt1 = d1["usage"]["prompt_tokens"]
        pt2 = d2["usage"]["prompt_tokens"]
        cal_reply = d2["choices"][0]["message"]["content"]
        assert pt1 < 40, f"Expected <40 prompt tokens for 1-word prompt, got {pt1}"
        assert pt1 != pt2, f"Expected distinct prompt tokens between models, got {pt1} vs {pt2}"
        return f"1-word prompt_tokens={pt1} (was 1045!), ornith-calendar reply={cal_reply!r}"
    check("8. Zero hidden tool-token bloat & distinct ornith-calendar identity", test_models_and_tokens)

    # 9. Input validation: empty messages, string messages (no 500 crash!), invalid role, negative max_tokens, temperature=5
    def test_bad_inputs():
        cases = [
            ("empty_messages", {"messages": []}),
            ("string_messages", {"messages": "hello"}),
            ("invalid_role", {"messages": [{"role": "banana", "content": "hi"}]}),
            ("negative_max_tokens", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": -5}),
            ("zero_max_tokens", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}),
            ("high_temperature", {"messages": [{"role": "user", "content": "hi"}], "temperature": 5}),
            ("negative_temperature", {"messages": [{"role": "user", "content": "hi"}], "temperature": -1}),
            ("bad_n", {"messages": [{"role": "user", "content": "hi"}], "n": 0}),
        ]
        for label, payload in cases:
            st, _, d, _ = http_req("POST", "/v1/chat/completions", payload)
            assert st == 400 and isinstance(d, dict) and "error" in d, f"Case {label} expected 400 JSON, got {st} {d!r}"
        return f"All {len(cases)} invalid payloads rejected with HTTP 400 JSON"
    check("9. Strict input validation (empty/string messages, bad role, max_tokens=-5, temp=5 -> 400)", test_bad_inputs)

    # 10. Context overflow (161,050 tokens) returns HTTP 400 (NEVER HTTP 200!)
    def test_context_overflow():
        huge = "word " * 161050
        st, _, d, _ = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{"role": "user", "content": huge}],
            "max_tokens": 10,
        }, timeout=30)
        assert st == 400 and isinstance(d, dict) and "error" in d, f"Expected 400 on 161K tokens, got {st} {d!r}"
        return f"status={st}, code={d['error'].get('code')}, msg={d['error']['message'][:70]}"
    check("10. Context overflow (161,050 tokens) returns HTTP 400 (not 200)", test_context_overflow)

    # 11. JSON mode (response_format: {"type": "json_object"})
    def test_json_mode():
        st, _, d, _ = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{"role": "user", "content": "Extract into JSON with keys name and role: Kwaku is a Principal Engineer."}],
            "response_format": {"type": "json_object"},
            "max_tokens": 80,
        })
        assert st == 200, f"Expected 200, got {st} {d}"
        content = d["choices"][0]["message"]["content"]
        parsed_json = json.loads(content)
        assert isinstance(parsed_json, dict), f"Expected JSON object, got {parsed_json}"
        return f"valid JSON returned: {json.dumps(parsed_json)}"
    check("11. JSON mode (response_format: json_object)", test_json_mode)

    # 12. Truncation at max_tokens: 16 returns finish_reason="length" and non-empty content
    def test_max_tokens_16():
        st, _, d, _ = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{"role": "user", "content": "Write a 200-word history of aviation."}],
            "max_tokens": 16,
        })
        assert st == 200, f"Expected 200, got {st} {d}"
        ch = d["choices"][0]
        fin = ch.get("finish_reason")
        txt = (ch.get("message") or {}).get("content") or ""
        assert fin == "length", f"Expected finish_reason='length', got {fin!r}"
        assert len(txt.strip()) > 0, f"Expected non-empty content at max_tokens=16, got {txt!r}"
        return f"finish_reason={fin!r}, content_len={len(txt)} chars ({txt[:40]!r}...)"
    check("12. max_tokens: 16 returns finish_reason='length' & non-empty content", test_max_tokens_16)

    # 13. Determinism with temperature: 0 and seed: 1
    def test_determinism():
        payload = {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{"role": "user", "content": "Write one sentence about Saturn's rings."}],
            "temperature": 0,
            "seed": 1,
            "max_tokens": 35,
        }
        st1, _, d1, _ = http_req("POST", "/v1/chat/completions", payload)
        st2, _, d2, _ = http_req("POST", "/v1/chat/completions", payload)
        assert st1 == 200 and st2 == 200
        c1 = d1["choices"][0]["message"]["content"]
        c2 = d2["choices"][0]["message"]["content"]
        assert c1 == c2, f"Mismatch:\n  1: {c1!r}\n  2: {c2!r}"
        return f"identical output={c1!r}"
    check("13. Deterministic output with temperature=0, seed=1", test_determinism)

    # 14. Multiple choices (n: 2)
    def test_n2():
        st, _, d, dt = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{"role": "user", "content": "Give a two-word greeting."}],
            "max_tokens": 12,
            "n": 2,
        })
        assert st == 200 and len(d.get("choices", [])) == 2, f"Expected 2 choices, got {st} {d}"
        assert d["choices"][0]["index"] == 0 and d["choices"][1]["index"] == 1
        return f"2 choices returned in {dt:.2f}s"
    check("14. n: 2 returns 2 choices cleanly", test_n2)

    # 15. Rate-limit headers present on responses
    def test_rl_headers():
        st, hdrs, _, _ = http_req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 4,
        })
        assert st == 200
        lim = hdrs.get("x-ratelimit-limit-requests")
        rem = hdrs.get("x-ratelimit-remaining-requests")
        assert lim and rem is not None, f"Missing rate-limit headers in {hdrs}"
        return f"limit={lim}, remaining={rem}"
    check("15. Rate-limit headers exposed on /v1 responses", test_rl_headers)

    # 16. Open-ended coding prompt returns complete runnable code in content (no sandbox narration trap!)
    def test_open_ended_coding():
        st, _, d, dt = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{
                "role": "user",
                "content": "Write a thread-safe TokenBucket class in Python with `consume(tokens)` and `refill()`. Include a short self-test.",
            }],
            "max_tokens": 650,
            "temperature": 0.2,
        })
        assert st == 200, f"Expected 200, got {st} {d}"
        content = d["choices"][0]["message"]["content"] or ""
        assert "class TokenBucket" in content and "def " in content, f"Missing code in content: {content[:200]!r}"
        return f"returned {len(content)} chars with `class TokenBucket` in {dt:.2f}s"
    check("16. Open-ended TokenBucket coding prompt returns code in content", test_open_ended_coding)

    # 17. Function calling (client-defined tool)
    def test_function_calling():
        st, _, d, _ = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [{"role": "user", "content": "What is the weather in Accra right now?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "max_tokens": 80,
        })
        assert st == 200, f"Got {st} {d}"
        ch = d["choices"][0]
        tcs = (ch.get("message") or {}).get("tool_calls") or []
        assert len(tcs) >= 1 and tcs[0]["function"]["name"] == "get_weather", f"Expected get_weather call, got {ch}"
        return f"tool_call={tcs[0]['function']}"
    check("17. Client function calling (get_weather)", test_function_calling)

    # 18. Short request latency while a long request is running concurrently (Slot reservation check)
    def test_short_during_long():
        def run_long():
            return http_req("POST", "/v1/chat/completions", {
                "model": "ornith-1.5-35b-uncensored",
                "messages": [{"role": "user", "content": "Write a detailed 400-word essay on distributed consensus algorithms."}],
                "max_tokens": 450,
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            long_fut = ex.submit(run_long)
            time.sleep(1.2)  # let the long request start occupying Slot 0
            st_short, _, d_short, dt_short = http_req("POST", "/v1/chat/completions", {
                "model": "ornith-1.5-35b-uncensored",
                "messages": [{"role": "user", "content": "Reply with one word: pong"}],
                "max_tokens": 5,
            })
            st_long, _, _, dt_long = long_fut.result()

        assert st_short == 200 and st_long == 200, f"short={st_short}, long={st_long}"
        assert dt_short < 5.0, f"Short request took {dt_short:.2f}s (expected <5s while long request ran for {dt_long:.2f}s)"
        return f"short_latency={dt_short:.2f}s while long_request ran ({dt_long:.2f}s)"
    check("18. Short request stays fast (<5s) while long essay generates", test_short_during_long)

    # 19. Load burst & Rate limiting: 20 concurrent requests trigger 429 with Retry-After and recover cleanly
    def test_load_burst():
        def one_req(i):
            st, hdrs, d, dt = http_req("POST", "/v1/chat/completions", {
                "model": "ornith-1.5-35b-uncensored",
                "messages": [{"role": "user", "content": f"Reply with one word: ok {i}"}],
                "max_tokens": 4,
            })
            return st, hdrs, dt

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            futs = [ex.submit(one_req, i) for i in range(20)]
            outs = [f.result() for f in futs]
        ok_count = sum(1 for st, _, _ in outs if st == 200)
        rl_count = sum(1 for st, _, _ in outs if st == 429)
        assert ok_count > 0 and rl_count > 0, f"Expected mix of 200 and 429 on 20 concurrent requests, got 200={ok_count}, 429={rl_count}"
        time.sleep(2.2)
        st_rec, _, _, dt_rec = http_req("POST", "/v1/chat/completions", {
            "model": "ornith-1.5-35b",
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 4,
        })
        assert st_rec == 200, f"Expected 200 after rate-limit recovery, got {st_rec}"
        return f"200={ok_count}, 429={rl_count}, recovered in {dt_rec:.2f}s"
    check("19. Concurrent load burst (20 simultaneous -> 429 + clean recovery)", test_load_burst)

    # 20. Model label fidelity in response
    def test_model_labels():
        for m in ("ornith-1.5-35b", "ornith-1.5-35b-uncensored", "ornith-1.5-35b-agent"):
            st, _, d, _ = http_req("POST", "/v1/chat/completions", {
                "model": m,
                "messages": [{"role": "user", "content": "Reply with one word: pong"}],
                "max_tokens": 8,
            })
            assert st == 200 and d.get("model") == m, f"Expected model={m}, got {st} {d}"
        return "all 3 model IDs echoed accurately"
    check("20. Model label fidelity across all 3 models", test_model_labels)

    print("\n" + "=" * 65)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"SUMMARY: {passed}/{len(results)} checks PASSED against {BASE_URL}")
    print("=" * 65)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(run_suite())
