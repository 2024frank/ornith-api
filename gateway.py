"""Unified Multi-Protocol API Gateway & Developer Platform for Ornith-1.5-35B-Uncensored.

Model Specification (ornith-1.5-35b-uncensored):
  - Architecture: ~35.5B params (Q8_0 quantization) + Vision & Video Multimodal Projector (BF16)
  - Engine: 131,072 context window, Flash Attention, unified KV cache, draft-mtp speculative decoding,
    2 parallel slots, 100% GPU offloaded.

Supported Wire Protocols:
  1. OpenAI Chat Completions: POST /v1/chat/completions
  2. OpenAI Completions:      POST /v1/completions
  3. OpenAI Responses (Codex):POST /v1/responses
  4. OpenAI Images API:       POST /v1/images/generations, POST /v1/images/edits
  5. Anthropic Messages (Claude Code CLI & SDK): POST /v1/messages, POST /v1/messages/count_tokens
  6. OpenAI Models & Chats:   GET /v1/models, GET /v1/chats, GET /v1/chats/<id>, POST /v1/feedback
  7. Developer Portal & Auth: GET /, POST /api/auth/request-code, POST /api/auth/verify-code,
                              /api/keys, /api/admin/users, /api/sandbox/run, /api/hardware, /api/logs
"""

import asyncio
import base64
import json
import os
import pathlib
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

import authdb
import chatdb
import server_tools

HERE = pathlib.Path(os.environ.get("ORNITH_HOME") or pathlib.Path(__file__).resolve().parent)
UI_CANDIDATES = [
    HERE / "ui" / "index.html",
    pathlib.Path(__file__).resolve().parent / "ui" / "index.html",
]
LOG = HERE / "access.log"
PUBLIC_URL_FILE = HERE / "public-url.txt"
UPSTREAM = "http://127.0.0.1:8081"
PRIMARY_MODEL = "ornith-1.5-35b-uncensored"
ALLOWED_EXPLICIT_PRESETS = {
    "ornith-1.5-35b-uncensored",
    "ornith-1.5-35b",
    "ornith-calendar",
    "ornith-1.5-35b-agent",
}
VALID_ROLES = {"system", "developer", "user", "assistant", "tool", "function"}
POST_ONLY_V1_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/responses",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/feedback",
}
GET_ONLY_V1_PATHS = {
    "/v1/models",
    "/v1/chats",
}
MAX_BODY = 35 * 1024 * 1024  # 35 MB for multimodal images & video clips

# Reserve 1 of the 2 GPU slots for interactive/short requests so a long generation never starves short requests.
_LONG_REQ_SEM = asyncio.Semaphore(1)

MODEL_SPEC = {
    "id": PRIMARY_MODEL,
    "display_name": "Ornith-1.5-35B-Uncensored",
    "quantization": "Q8_0",
    "parameters": "35.5B",
    "modalities": ["text", "image", "video"],
    "context_window": 131072,
    "trained_context": 262144,
    "flash_attention": True,
    "kv_unified": True,
    "speculative_decoding": "draft-mtp",
    "parallel_slots": 2,
    "storage_mode": "100% Local Encrypted SQLite",
    "gateway_public": "https://kusi-qwen-api.vercel.app",
}

MODEL_IDENTITY_NOTES = {
    "ornith-1.5-35b-uncensored": "You are Ornith-1.5-35B-Uncensored by Ornith AI.",
    "ornith-1.5-35b": "You are Ornith-1.5-35B by Ornith AI.",
    "ornith-calendar": (
        "You are Ornith-Calendar (ornith-calendar) by Ornith AI, a specialized model fine-tuned for "
        "calendar scheduling, event extraction, and concise time-management assistance."
    ),
    "ornith-1.5-35b-agent": "You are Ornith-1.5-35B-Agent by Ornith AI with sandbox and tool execution.",
}

MODEL_CATALOG = [
    {
        "id": "ornith-1.5-35b-uncensored",
        "object": "model",
        "created": 1740000000,
        "owned_by": "ornith-ai",
        "status": "loaded",
        "context_window": 131072,
        "modalities": ["text", "image", "video"],
        "description": "Primary uncensored 35.5B multimodal model (Q8_0 + BF16 vision/video projector, 131K context).",
        "spec": MODEL_SPEC,
    },
    {
        "id": "ornith-calendar",
        "object": "model",
        "created": 1740000100,
        "owned_by": "ornith-ai",
        "status": "loaded",
        "context_window": 131072,
        "modalities": ["text"],
        "description": "Ornith 35.5B fine-tuned with AI Calendar LoRA and control-vector steering for scheduling and event extraction.",
    },
    {
        "id": "ornith-1.5-35b",
        "object": "model",
        "created": 1740000200,
        "owned_by": "ornith-ai",
        "status": "loaded",
        "context_window": 131072,
        "modalities": ["text", "image", "video"],
        "description": "Standard Ornith-1.5-35B multimodal model.",
    },
    {
        "id": "ornith-1.5-35b-agent",
        "object": "model",
        "created": 1740000300,
        "owned_by": "ornith-ai",
        "status": "loaded",
        "context_window": 131072,
        "modalities": ["text", "image", "video"],
        "description": "Opt-in autonomous agent preset with server-side Python/Bash sandbox, web search, and image synthesis tools.",
    },
]


def resolve_and_validate_model(requested: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Validate requested model name.
    Returns (client_model_name, upstream_preset_name, error_message).
    - Omitted or empty `model` defaults to PRIMARY_MODEL.
    - Unknown `model` returns (None, None, error_message) so caller returns 404 model_not_found.
    - `ornith-1.5-35b` and `ornith-1.5-35b-agent` route to the resident `ornith-1.5-35b-uncensored` VRAM slot
      so llama-server never evicts `ornith-1.5-35b-uncensored` or `ornith-calendar` from VRAM.
    """
    if requested is None or (isinstance(requested, str) and not requested.strip()):
        return PRIMARY_MODEL, PRIMARY_MODEL, None
    if not isinstance(requested, str):
        return None, None, "Invalid 'model' parameter: must be a string."
    s = requested.strip()
    if s not in ALLOWED_EXPLICIT_PRESETS:
        return None, None, f"The model '{s}' does not exist or you do not have access to it."
    upstream_model = "ornith-calendar" if s == "ornith-calendar" else PRIMARY_MODEL
    return s, upstream_model, None


def is_long_request(body: Dict[str, Any], server_tools_enabled: bool = False) -> bool:
    """Classify whether a request is expected to run long so we reserve 1 of the 2 GPU slots for fast requests."""
    if server_tools_enabled:
        return True
    max_tok = body.get("max_tokens") or body.get("max_completion_tokens")
    if isinstance(max_tok, int) and not isinstance(max_tok, bool):
        if max_tok <= 300:
            return False
        if max_tok > 500:
            return True
    # Inspect prompt length and keywords when max_tokens is omitted or moderate
    texts: List[str] = []
    msgs = body.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str):
                    texts.append(c)
    elif isinstance(body.get("prompt"), str):
        texts.append(body["prompt"])
    combined = " ".join(texts)
    if len(combined) > 1200:
        return True
    if re.search(r"\b(essay|article|comprehensive|detailed|1,?000|1,?500|2,?000|words|chapter|story|tutorial)\b", combined, re.IGNORECASE):
        return True
    return False


def validate_chat_or_completion_input(path: str, body: Any) -> Optional[Tuple[int, str, str, Optional[str], Optional[str]]]:
    """Validate OpenAI / Anthropic request payload before forwarding to upstream.
    Returns (http_status, message, err_type, param, code) on error, or None if valid.
    """
    if not isinstance(body, dict):
        return (400, "Request body must be a JSON object.", "invalid_request_error", None, "invalid_json")

    # 1. Validate messages for chat endpoints
    if path in ("/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or len(msgs) == 0:
            return (
                400,
                "'messages' must be a non-empty array of message objects.",
                "invalid_request_error",
                "messages",
                "invalid_messages",
            )
        for idx, m in enumerate(msgs):
            if not isinstance(m, dict):
                return (
                    400,
                    f"messages[{idx}] must be an object with 'role' and 'content'.",
                    "invalid_request_error",
                    f"messages[{idx}]",
                    "invalid_message_format",
                )
            role = m.get("role")
            if not isinstance(role, str) or role not in VALID_ROLES:
                return (
                    400,
                    f"Invalid role '{role}' at messages[{idx}]. Must be one of: {', '.join(sorted(VALID_ROLES))}.",
                    "invalid_request_error",
                    f"messages[{idx}].role",
                    "invalid_role",
                )
            content = m.get("content")
            has_tool_calls = bool(m.get("tool_calls") or m.get("function_call"))
            if content is None and not has_tool_calls and role != "assistant":
                return (
                    400,
                    f"messages[{idx}].content is required for role '{role}'.",
                    "invalid_request_error",
                    f"messages[{idx}].content",
                    "missing_content",
                )
            if content is not None and not isinstance(content, (str, list)):
                return (
                    400,
                    f"messages[{idx}].content must be a string or array of content parts.",
                    "invalid_request_error",
                    f"messages[{idx}].content",
                    "invalid_content_type",
                )

    if path == "/v1/completions":
        prompt = body.get("prompt")
        if prompt is None or not isinstance(prompt, (str, list)) or (isinstance(prompt, str) and not prompt):
            return (
                400,
                "'prompt' must be a non-empty string or array.",
                "invalid_request_error",
                "prompt",
                "invalid_prompt",
            )

    # 2. Validate max_tokens / max_completion_tokens
    for tok_key in ("max_tokens", "max_completion_tokens"):
        if tok_key in body and body[tok_key] is not None:
            val = body[tok_key]
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                return (
                    400,
                    f"'{tok_key}' must be an integer >= 1 (got {val!r}).",
                    "invalid_request_error",
                    tok_key,
                    "invalid_max_tokens",
                )
            if val > 131072:
                return (
                    400,
                    f"'{tok_key}' ({val}) exceeds the maximum context window (131072).",
                    "invalid_request_error",
                    tok_key,
                    "context_length_exceeded",
                )

    # 3. Validate temperature (0 to 2)
    if "temperature" in body and body["temperature"] is not None:
        temp = body["temperature"]
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
            return (
                400,
                f"'temperature' must be a number between 0 and 2 (got {temp!r}).",
                "invalid_request_error",
                "temperature",
                "invalid_temperature",
            )

    # 4. Validate top_p (0 to 1)
    if "top_p" in body and body["top_p"] is not None:
        top_p = body["top_p"]
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or top_p < 0.0 or top_p > 1.0:
            return (
                400,
                f"'top_p' must be a number between 0 and 1 (got {top_p!r}).",
                "invalid_request_error",
                "top_p",
                "invalid_top_p",
            )

    # 5. Validate n (1 to 8)
    if "n" in body and body["n"] is not None:
        n_val = body["n"]
        if isinstance(n_val, bool) or not isinstance(n_val, int) or n_val < 1 or n_val > 8:
            return (
                400,
                f"'n' must be an integer between 1 and 8 (got {n_val!r}).",
                "invalid_request_error",
                "n",
                "invalid_n",
            )

    # 6. Validate response_format
    if "response_format" in body and body["response_format"] is not None:
        rf = body["response_format"]
        if not isinstance(rf, dict) or rf.get("type") not in ("text", "json_object", "json_schema"):
            return (
                400,
                "'response_format' must be an object with 'type' set to 'text', 'json_object', or 'json_schema'.",
                "invalid_request_error",
                "response_format",
                "invalid_response_format",
            )
        if rf.get("type") == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or not isinstance(js.get("schema"), dict):
                return (
                    400,
                    "response_format of type 'json_schema' requires a 'json_schema.schema' object.",
                    "invalid_request_error",
                    "response_format.json_schema",
                    "invalid_json_schema",
                )

    return None


def parse_upstream_error(status: int, raw_bytes: bytes) -> Tuple[int, Dict[str, Any]]:
    """Map upstream llama-server error payload into a clean OpenAI-style error dict and proper HTTP status."""
    mapped_status = status if 400 <= status < 600 else 500
    msg = "Upstream model server error."
    err_type = "invalid_request_error" if 400 <= mapped_status < 500 else "server_error"
    code: Optional[str] = None
    param: Optional[str] = None

    try:
        parsed = json.loads(raw_bytes.decode("utf-8", "replace"))
        if isinstance(parsed, dict):
            err_obj = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
            if isinstance(err_obj.get("message"), str) and err_obj["message"].strip():
                msg = err_obj["message"].strip()
            if isinstance(err_obj.get("type"), str) and err_obj["type"].strip():
                err_type = err_obj["type"].strip()
            if "exceed" in msg.lower() and "context" in msg.lower():
                mapped_status = 400
                err_type = "invalid_request_error"
                code = "context_length_exceeded"
                param = "messages"
            elif "grammar" in msg.lower() or "json" in msg.lower():
                mapped_status = 400
                err_type = "invalid_request_error"
                code = "invalid_response_format"
            elif "chat template" in msg.lower() or "system message" in msg.lower():
                mapped_status = 400
                err_type = "invalid_request_error"
                code = "invalid_messages"
    except Exception:
        txt = raw_bytes.decode("utf-8", "replace").strip()
        if txt and not txt.startswith("<"):
            msg = txt[:400]

    return mapped_status, {
        "error": {
            "message": msg,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


def log_access(**kw) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}) + "\n")
    except Exception:
        pass


def client_ip(req: web.Request) -> str:
    return (
        req.headers.get("CF-Connecting-IP")
        or req.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or req.remote
        or "127.0.0.1"
    )


def cors_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    base = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-Key, Anthropic-Version, Anthropic-Beta, X-Conversation-Id",
        "Access-Control-Expose-Headers": (
            "X-Chat-Id, X-Ornith-Model, Retry-After, Allow, "
            "X-RateLimit-Limit-Requests, X-RateLimit-Remaining-Requests, X-RateLimit-Reset-Requests"
        ),
    }
    if extra:
        base.update(extra)
    return base


def json_err(
    status: int,
    msg: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": msg,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
        status=status,
        headers=cors_headers(extra_headers),
    )


def extract_session_token(req: web.Request) -> Optional[str]:
    cookie_tok = req.cookies.get("ornith_session")
    if cookie_tok and cookie_tok.startswith("ornith_sess_"):
        return cookie_tok
    auth = req.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        if tok.startswith("ornith_sess_"):
            return tok
    x_key = req.headers.get("x-api-key", "").strip()
    if x_key.startswith("ornith_sess_"):
        return x_key
    return None


def extract_api_credential(req: web.Request) -> Optional[Dict[str, Any]]:
    auth = req.headers.get("Authorization", "")
    given = ""
    if auth.lower().startswith("bearer "):
        given = auth[7:].strip()
    if not given:
        given = req.headers.get("x-api-key", "").strip()
    if not given:
        given = req.cookies.get("ornith_session", "").strip()
    return authdb.verify_api_key(given)


# ── Hardware Telemetry & Always-On Watchdog ──────────────────────────────

async def query_hardware_state(http_session: aiohttp.ClientSession) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "online": False,
        "model_status": "unloaded",
        "slots": [],
        "spec": MODEL_SPEC,
        "public_url": "https://kusi-qwen-api.vercel.app",
        "storage_mode": "local",
    }
    try:
        async with http_session.get(f"{UPSTREAM}/v1/models", timeout=aiohttp.ClientTimeout(total=3)) as r:
            if r.status == 200:
                data = await r.json()
                state["online"] = True
                for m in data.get("data", []):
                    if m.get("id") == PRIMARY_MODEL:
                        st = m.get("status", {})
                        state["model_status"] = st.get("value", "unknown")
                        break
    except Exception:
        state["online"] = False

    if state["online"] and state["model_status"] == "loaded":
        try:
            async with http_session.get(
                f"{UPSTREAM}/slots?model={PRIMARY_MODEL}",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as sr:
                if sr.status == 200:
                    raw_slots = await sr.json()
                    slots_summary = []
                    for s in raw_slots if isinstance(raw_slots, list) else []:
                        nt = (s.get("next_token") or [{}])[0] if isinstance(s.get("next_token"), list) and s.get("next_token") else {}
                        slots_summary.append(
                            {
                                "id": s.get("id", 0),
                                "is_processing": bool(s.get("is_processing")),
                                "n_ctx": s.get("n_ctx", 65536),
                                "n_prompt_tokens_processed": s.get("n_prompt_tokens_processed", 0),
                                "n_decoded": nt.get("n_decoded", 0),
                                "n_remain": nt.get("n_remain", 0),
                            }
                        )
                    state["slots"] = slots_summary
        except Exception:
            pass

    return state


async def always_on_watchdog(app: web.Application) -> None:
    """Ensure ornith-1.5-35b-uncensored stays loaded and responsive 24/7."""
    await asyncio.sleep(5)
    while True:
        try:
            hw = await query_hardware_state(app["session"])
            if not hw["online"]:
                proc = await asyncio.create_subprocess_exec(
                    "systemctl", "--user", "start", "llama-qwen3.6.service",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            elif hw["model_status"] != "loaded":
                async with app["session"].post(
                    f"{UPSTREAM}/v1/chat/completions",
                    json={
                        "model": PRIMARY_MODEL,
                        "messages": [{"role": "user", "content": "ok"}],
                        "max_tokens": 1,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as _:
                    pass
        except asyncio.CancelledError:
            break
        except Exception:
            pass
        await asyncio.sleep(45)


_HOME_ESC = re.escape(str(pathlib.Path.home()))
_PROMPT_EXFIL_RE = re.compile(
    r"(?:fetch|read|dump|find|list|show|print|get|cat|steal|exfiltrate|search|open|display)\b[^\n]{0,100}"
    r"(?:\.env(?:\.[a-zA-Z0-9_-]+|\b)|RESEND_API_KEY|" + _HOME_ESC + r"\b|ornith_platform\.db|keys\.txt|/etc/shadow)",
    re.IGNORECASE,
)


def _check_inference_host_guardrail(body: Dict[str, Any]) -> Optional[str]:
    """Protect host .env files, credentials, and local filesystem from API prompt exfiltration or file:// SSRF."""
    raw_texts: List[str] = []

    def _scan_item(item: Any) -> Optional[str]:
        if isinstance(item, str):
            raw_texts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "image_url":
                u = (item.get("image_url") or {}).get("url", "")
                if isinstance(u, str) and u.strip():
                    low = u.strip().lower()
                    if low.startswith("file:") or "127.0.0.1" in low or "localhost" in low:
                        return "[Host Security Guardrail] Local file:// or loopback image_url references are blocked."
            for v in item.values():
                err = _scan_item(v)
                if err:
                    return err
        elif isinstance(item, list):
            for sub in item:
                err = _scan_item(sub)
                if err:
                    return err
        return None

    for key in ("messages", "input", "prompt"):
        if key in body:
            err = _scan_item(body[key])
            if err:
                return err

    combined_user_text = "\n".join(raw_texts)
    if _PROMPT_EXFIL_RE.search(combined_user_text):
        return (
            "[Host Security Guardrail] Requests to fetch, read, or dump host `.env` files, server credentials, "
            "or protected host filesystem paths are blocked to protect the host environment. "
            "All other uncensored coding, reasoning, creative, and analytical requests are fully permitted."
        )
    return None


def _normalize_responses_payload(payload: Dict[str, Any], upstream_model: str) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, str]], bool]:
    ns_map: Dict[str, Tuple[str, str]] = {}
    custom_apply_patch = False
    payload["model"] = upstream_model

    if "tools" in payload and isinstance(payload["tools"], list):
        flat_tools = []
        for t in payload["tools"]:
            if not isinstance(t, dict):
                continue
            t_type = t.get("type")
            if t_type == "function":
                flat_tools.append(t)
            elif t_type == "namespace":
                ns_name = t.get("name", "")
                for inner in t.get("tools", []):
                    if isinstance(inner, dict) and inner.get("type") == "function":
                        orig_name = inner.get("name", "")
                        flat_name = f"{ns_name}__{orig_name}"
                        ns_map[flat_name] = (ns_name, orig_name)
                        fn_copy = dict(inner)
                        fn_copy["name"] = flat_name
                        flat_tools.append(fn_copy)
            elif t_type == "custom" and t.get("name") == "apply_patch":
                custom_apply_patch = True
                flat_tools.append(
                    {
                        "type": "function",
                        "name": "apply_patch",
                        "description": "Apply a file patch using unified patch format (*** Begin Patch ... *** End Patch).",
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                            "required": ["input"],
                        },
                    }
                )
        payload["tools"] = flat_tools

    if isinstance(payload.get("input"), list):
        sys_texts, kept = [], []
        for item in payload["input"]:
            if isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") in ("system", "developer"):
                c = item.get("content")
                text = c if isinstance(c, str) else "\n".join(
                    p.get("text", "") for p in (c or []) if isinstance(p, dict)
                )
                if text.strip():
                    sys_texts.append(text)
            else:
                kept.append(item)
        if sys_texts:
            payload["instructions"] = "\n\n".join(filter(None, [payload.get("instructions") or ""] + sys_texts))
            payload["input"] = kept

    return payload, ns_map, custom_apply_patch


# ── Main HTTP Handler ────────────────────────────────────────────────────

async def handle(req: web.Request) -> web.StreamResponse:
    try:
        return await _handle_inner(req)
    except web.HTTPException as http_ex:
        return json_err(http_ex.status, http_ex.reason or "HTTP error.", "invalid_request_error")
    except Exception as exc:
        log_access(path=req.path, status=500, error=str(exc)[:200])
        return json_err(500, "Internal gateway error while processing request.", "server_error", code="internal_error")


async def _handle_inner(req: web.Request) -> web.StreamResponse:
    if req.method == "OPTIONS":
        return web.Response(status=204, headers=cors_headers())

    path = req.path.rstrip("/") or "/"
    ip = client_ip(req)
    loop = asyncio.get_running_loop()

    # 1. Developer Console UI & Public Docs (/docs, /cli)
    if req.method == "GET" and path in ("/", "/index.html", "/console", "/docs", "/cli", "/playground", "/sandbox"):
        for ui_path in UI_CANDIDATES:
            if ui_path.exists():
                return web.FileResponse(
                    ui_path,
                    headers={"Cache-Control": "no-cache", **cors_headers()},
                )
        return json_err(404, "Developer Console UI file not found.")

    # 1b. CLI Installer & Download Endpoints (/install.sh & /cli/ornith.py)
    if req.method == "GET" and path == "/cli/ornith.py":
        cli_path = HERE / "ornith_sdk.py"
        if cli_path.exists():
            return web.FileResponse(
                cli_path,
                headers={"Content-Type": "text/x-python; charset=utf-8", "Cache-Control": "no-cache", **cors_headers()},
            )
        return json_err(404, "CLI script not found.")

    if req.method == "GET" and path == "/install.sh":
        host_hdr = req.headers.get("Host", "")
        if host_hdr.startswith("127.0.0.1") or host_hdr.startswith("localhost"):
            origin = f"http://{host_hdr}"
        else:
            origin = "https://kusi-qwen-api.vercel.app"
        install_script = f"""#!/usr/bin/env bash
set -euo pipefail

echo "Installing Ornith-1.5-35B-Uncensored CLI & Coding Agent Suite..."
mkdir -p "$HOME/.local/bin" "$HOME/.config/ornith"
chmod 700 "$HOME/.config/ornith"

rm -f "$HOME/.local/bin/ornith" "$HOME/.local/bin/onith"
curl -fsSL "{origin}/cli/ornith.py" -o "$HOME/.local/bin/ornith"
chmod +x "$HOME/.local/bin/ornith"
ln -sf "$HOME/.local/bin/ornith" "$HOME/.local/bin/onith"

if [ -d "$HOME/.local/node/bin" ] && [ -w "$HOME/.local/node/bin" ]; then
  ln -sf "$HOME/.local/bin/ornith" "$HOME/.local/node/bin/ornith"
  ln -sf "$HOME/.local/bin/ornith" "$HOME/.local/node/bin/onith"
fi

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  if [ -f "$rc" ] && ! grep -q '\\.local/bin' "$rc"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
  fi
done
export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "✅ Ornith CLI installed to ~/.local/bin/ornith"
echo ""
echo "Next steps:"
echo "  1. Ensure ~/.local/bin is active in your current terminal tab:"
echo "     export PATH=\\"\$HOME/.local/bin:\$PATH\\""
""".replace("\\$", "$") + """echo ""
echo "  2. Authenticate your account (via 6-digit Resend code or API key):"
echo "     ornith login"
echo ""
echo "  3. Start interactive coding agent or run a prompt:"
echo "     ornith code"
echo "     ornith \\"Explain this repository\\""
"""
        return web.Response(
            text=install_script,
            content_type="text/x-shellscript",
            headers={"Cache-Control": "no-cache", **cors_headers()},
        )

    # 2. Public generated image files
    if req.method == "GET" and re.fullmatch(r"/files/[0-9a-f]{32}\.png", path):
        p = server_tools.GENERATED / path.rsplit("/", 1)[1]
        if p.exists():
            return web.FileResponse(
                p,
                headers={"Cache-Control": "public, max-age=31536000, immutable", **cors_headers()},
            )
        return json_err(404, "No such generated image.")

    # 3. Health & Hardware status
    if req.method == "GET" and path in ("/health", "/api/hardware"):
        hw = await query_hardware_state(req.app["session"])
        is_ready = hw["online"] and hw["model_status"] == "loaded"
        return web.json_response(
            {
                "ok": is_ready,
                "status": "healthy" if is_ready else "degraded",
                "model": PRIMARY_MODEL,
                "heretic_model": PRIMARY_MODEL,
                "heretic_online": hw["online"],
                "hardware": hw,
            },
            status=200 if is_ready else 503,
            headers=cors_headers({"Retry-After": "5"} if not is_ready else None),
        )

    # 4. Authentication Endpoints (Resend 6-digit OTP)
    if req.method == "POST" and path == "/api/auth/request-code":
        try:
            body = await req.json()
        except Exception:
            return json_err(400, "JSON body required.")
        ok, status, data = await loop.run_in_executor(
            None, authdb.request_login_otp, body.get("email", ""), ip
        )
        if not ok:
            return json_err(status, data.get("error", "Failed to send code."))
        return web.json_response(data, status=status, headers=cors_headers())

    if req.method == "POST" and path == "/api/auth/verify-code":
        try:
            body = await req.json()
        except Exception:
            return json_err(400, "JSON body required.")
        ua = req.headers.get("User-Agent", "")
        ok, status, data = await loop.run_in_executor(
            None,
            authdb.verify_login_otp,
            body.get("email", ""),
            body.get("code", ""),
            ip,
            ua,
        )
        if not ok:
            return json_err(status, data.get("error", "Verification failed."))
        if body.get("mint_cli_key"):
            cli_label = re.sub(r"[^a-zA-Z0-9_.-]", "-", str(body.get("client_name") or "ornith-cli"))[:40]
            minted = await loop.run_in_executor(
                None,
                authdb.create_api_key,
                data["user"]["email"],
                cli_label,
                int(data["user"].get("rate_limit_rpm") or 120),
                True,
                bool(data["user"].get("allow_command_exec", True)),
                True,
            )
            data["api_key"] = minted["key"]
            data["key_prefix"] = minted["key_prefix"]
        resp = web.json_response(data, status=200, headers=cors_headers())
        resp.set_cookie(
            "ornith_session",
            data["session_token"],
            max_age=authdb.SESSION_TTL_SECONDS,
            path="/",
            httponly=True,
            samesite="Lax",
        )
        return resp

    if req.method == "POST" and path == "/api/auth/logout":
        tok = extract_session_token(req)
        await loop.run_in_executor(None, authdb.revoke_session, tok)
        resp = web.json_response({"ok": True}, headers=cors_headers())
        resp.del_cookie("ornith_session", path="/")
        return resp

    if req.method == "GET" and path == "/api/auth/me":
        tok = extract_session_token(req)
        user = await loop.run_in_executor(None, authdb.get_session_user, tok)
        if not user:
            cred = extract_api_credential(req)
            if cred:
                user = {
                    "email": cred["user_email"],
                    "name": cred["key_name"],
                    "role": cred["role"],
                    "status": "active",
                    "rate_limit_rpm": cred["rate_limit_rpm"],
                    "allow_command_exec": cred["allow_command"],
                }
        hw = await query_hardware_state(req.app["session"])
        if not user:
            return web.json_response(
                {"authenticated": False, "hardware": hw, "spec": MODEL_SPEC},
                headers=cors_headers(),
            )
        is_admin = user["role"] == "admin"
        keys = await loop.run_in_executor(None, authdb.list_api_keys, user["email"], is_admin)
        logs = await loop.run_in_executor(
            None, authdb.list_audit_logs, 60, None if is_admin else user["email"]
        )
        users_list = await loop.run_in_executor(None, authdb.list_users) if is_admin else []
        presets_list = await loop.run_in_executor(None, authdb.list_presets, user["email"])
        return web.json_response(
            {
                "authenticated": True,
                "user": user,
                "hardware": hw,
                "spec": MODEL_SPEC,
                "keys": keys,
                "logs": logs,
                "users": users_list,
                "presets": presets_list,
                "tools": [t["function"] for t in server_tools.TOOLS],
            },
            headers=cors_headers(),
        )

    # 5. Authenticated Developer Console APIs (/api/*)
    if path.startswith("/api/"):
        tok = extract_session_token(req)
        sess_user = await loop.run_in_executor(None, authdb.get_session_user, tok)
        api_cred = extract_api_credential(req) if not sess_user else None

        if not sess_user and not api_cred:
            return json_err(401, "Authentication required. Sign in with Resend OTP or pass an API key.", "authentication_error")

        actor_email = sess_user["email"] if sess_user else api_cred["user_email"]
        is_admin = (sess_user["role"] == "admin") if sess_user else (api_cred["role"] == "admin")
        allow_cmd = bool(sess_user["allow_command_exec"]) if sess_user else bool(api_cred["allow_command"])

        if path == "/api/presets":
            if req.method == "GET":
                presets_list = await loop.run_in_executor(None, authdb.list_presets, actor_email)
                return web.json_response({"presets": presets_list}, headers=cors_headers())
            if req.method == "POST":
                body = await req.json()
                saved = await loop.run_in_executor(
                    None,
                    authdb.save_preset,
                    actor_email,
                    body.get("name", "Custom Preset"),
                    body.get("description", ""),
                    body.get("system_prompt", ""),
                    float(body.get("temperature", 0.6)),
                    float(body.get("top_p", 0.95)),
                    int(body.get("max_tokens", 4096)),
                    bool(body.get("enable_thinking", True)),
                    body.get("tools", []),
                )
                return web.json_response({"ok": True, "preset": saved}, status=201, headers=cors_headers())

        m_preset_del = re.fullmatch(r"/api/presets/([a-zA-Z0-9_.-]+)", path)
        if m_preset_del and req.method == "DELETE":
            pid = m_preset_del.group(1)
            ok = await loop.run_in_executor(None, authdb.delete_preset, pid, actor_email, is_admin)
            return web.json_response({"ok": ok}, status=200 if ok else 404, headers=cors_headers())

        if path == "/api/keys":
            if req.method == "GET":
                keys = await loop.run_in_executor(None, authdb.list_api_keys, actor_email, is_admin)
                return web.json_response({"keys": keys}, headers=cors_headers())
            if req.method == "POST":
                body = await req.json()
                created = await loop.run_in_executor(
                    None,
                    authdb.create_api_key,
                    body.get("user_email") if (is_admin and body.get("user_email")) else actor_email,
                    body.get("name", "api-key"),
                    int(body.get("rate_limit_rpm") or 60),
                    bool(body.get("allow_tools", True)),
                    bool(body.get("allow_command", True)) and allow_cmd,
                    bool(body.get("train_ok", True)),
                )
                authdb.record_audit(actor_email, "key", f"mint_key:{created['name']}", 200, ip=ip)
                return web.json_response(created, status=201, headers=cors_headers())

        m_key_del = re.fullmatch(r"/api/keys/([0-9a-fA-F]+)", path)
        if m_key_del and req.method == "DELETE":
            kid = m_key_del.group(1)
            ok = await loop.run_in_executor(None, authdb.revoke_api_key, kid, actor_email, is_admin)
            authdb.record_audit(actor_email, "key", f"revoke_key:{kid}", 200 if ok else 404, ip=ip)
            return web.json_response({"ok": ok}, status=200 if ok else 404, headers=cors_headers())

        if path == "/api/logs" and req.method == "GET":
            logs = await loop.run_in_executor(
                None, authdb.list_audit_logs, int(req.query.get("limit", 100)), None if is_admin else actor_email
            )
            return web.json_response({"logs": logs}, headers=cors_headers())

        if path == "/api/files" and req.method == "GET":
            files_out = []
            if server_tools.GENERATED.exists():
                for p in sorted(server_tools.GENERATED.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:60]:
                    st = p.stat()
                    files_out.append({
                        "name": p.name,
                        "url": f"/files/{p.name}",
                        "public_url": f"https://kusi-qwen-api.vercel.app/files/{p.name}",
                        "size_bytes": st.st_size,
                        "created_at": st.st_mtime,
                        "kind": "generated_image",
                    })
            return web.json_response({"files": files_out}, headers=cors_headers())

        if path == "/api/usage" and req.method == "GET":
            keys = await loop.run_in_executor(None, authdb.list_api_keys, actor_email, is_admin)
            logs = await loop.run_in_executor(None, authdb.list_audit_logs, 250, None if is_admin else actor_email)
            total_reqs = sum(int(k.get("total_requests") or 0) for k in keys)
            total_in = sum(int(k.get("total_tokens_in") or 0) for k in keys)
            total_out = sum(int(k.get("total_tokens_out") or 0) for k in keys)
            by_cat: Dict[str, int] = {}
            latencies: List[float] = []
            for l in logs:
                cat = l.get("category", "other")
                by_cat[cat] = by_cat.get(cat, 0) + 1
                if l.get("latency_ms", 0) > 0:
                    latencies.append(float(l["latency_ms"]))
            avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
            return web.json_response({
                "total_requests": total_reqs,
                "total_tokens_in": total_in,
                "total_tokens_out": total_out,
                "total_tokens": total_in + total_out,
                "avg_latency_ms": avg_lat,
                "by_category": by_cat,
                "active_keys": sum(1 for k in keys if not k.get("revoked")),
            }, headers=cors_headers())

        if path == "/api/sandbox/run" and req.method == "POST":
            body = await req.json()
            mode = body.get("mode", "sandbox")
            if mode == "command":
                if not allow_cmd:
                    return json_err(403, "Host command execution is not enabled for this account.")
                res = await loop.run_in_executor(
                    None,
                    server_tools.execute_command_sync,
                    str(body.get("command") or body.get("code") or ""),
                    body.get("cwd"),
                    int(body.get("timeout") or 30),
                )
                authdb.record_audit(
                    actor_email,
                    "command",
                    str(body.get("command") or body.get("code") or "")[:80],
                    200 if res["exit_code"] == 0 else 500,
                    latency_ms=res["duration_ms"],
                    ip=ip,
                )
                return web.json_response(res, headers=cors_headers())
            else:
                res = await loop.run_in_executor(
                    None,
                    server_tools.execute_sandbox_sync,
                    str(body.get("language", "python")),
                    str(body.get("code", "")),
                    int(body.get("timeout") or 20),
                    str(body.get("stdin", "")),
                )
                authdb.record_audit(
                    actor_email,
                    "sandbox",
                    f"{res['language']}:exit={res['exit_code']}",
                    200 if res["exit_code"] == 0 else 500,
                    latency_ms=res["duration_ms"],
                    ip=ip,
                )
                return web.json_response(res, headers=cors_headers())

        if path == "/api/media/extract-frames" and req.method == "POST":
            body = await req.json()
            b64_data = str(body.get("video_base64") or "")
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            raw_video = base64.b64decode(b64_data)
            ext = str(body.get("ext") or ".mp4")
            max_frames = int(body.get("max_frames") or 4)
            frames = await loop.run_in_executor(
                None, server_tools.extract_video_keyframes, raw_video, ext, max_frames
            )
            return web.json_response({"frames": frames, "count": len(frames)}, headers=cors_headers())

        if path == "/api/tools/run" and req.method == "POST":
            body = await req.json()
            t_name = str(body.get("tool") or "")
            t_args = body.get("args") or {}
            generated: List[str] = []
            res_text, status_line = await server_tools.run(
                t_name, t_args, body.get("messages") or [], generated, None, allow_command=allow_cmd
            )
            authdb.record_audit(actor_email, "tool", f"{t_name}", 200, ip=ip, detail=t_args)
            return web.json_response(
                {"ok": True, "tool": t_name, "status_line": status_line, "result": res_text, "generated_files": generated},
                headers=cors_headers(),
            )

        # Admin User Directory Endpoints
        if path.startswith("/api/admin/users"):
            if not is_admin:
                return json_err(403, "Administrator privileges required.")
            if req.method == "GET" and path == "/api/admin/users":
                users_list = await loop.run_in_executor(None, authdb.list_users)
                return web.json_response({"users": users_list}, headers=cors_headers())
            if req.method == "POST" and path == "/api/admin/users":
                body = await req.json()
                ok, msg, user_row = await loop.run_in_executor(
                    None,
                    authdb.add_or_update_user,
                    body.get("email", ""),
                    body.get("name", ""),
                    body.get("role", "developer"),
                    body.get("status", "active"),
                    int(body.get("rate_limit_rpm") or 60),
                    bool(body.get("allow_command_exec", True)),
                    body.get("notes", ""),
                    actor_email,
                    bool(body.get("send_invite", True)),
                )
                if not ok:
                    return json_err(400, msg)
                authdb.record_audit(actor_email, "admin", f"upsert_user:{user_row['email']}", 200, ip=ip, detail=user_row)
                return web.json_response({"ok": True, "message": msg, "user": user_row}, headers=cors_headers())

            m_usr = re.fullmatch(r"/api/admin/users/([^/]+)", path)
            if m_usr:
                target_email = m_usr.group(1)
                if req.method == "PATCH":
                    body = await req.json()
                    ok, msg, user_row = await loop.run_in_executor(
                        None,
                        authdb.add_or_update_user,
                        target_email,
                        body.get("name", ""),
                        body.get("role", "developer"),
                        body.get("status", "active"),
                        int(body.get("rate_limit_rpm") or 60),
                        bool(body.get("allow_command_exec", True)),
                        body.get("notes", ""),
                        actor_email,
                        False,
                    )
                    if not ok:
                        return json_err(400, msg)
                    authdb.record_audit(actor_email, "admin", f"update_user:{target_email}", 200, ip=ip)
                    return web.json_response({"ok": True, "user": user_row}, headers=cors_headers())
                if req.method == "DELETE":
                    ok, msg = await loop.run_in_executor(None, authdb.delete_user, target_email)
                    if not ok:
                        return json_err(400, msg)
                    authdb.record_audit(actor_email, "admin", f"delete_user:{target_email}", 200, ip=ip)
                    return web.json_response({"ok": True, "message": msg}, headers=cors_headers())

        return json_err(404, f"Unknown API endpoint: {path}")

    # 6. Check Method Not Allowed (405) for known /v1 endpoints BEFORE or AFTER auth
    if path in POST_ONLY_V1_PATHS and req.method != "POST":
        return json_err(
            405,
            f"Method {req.method} not allowed for {path}. Use POST.",
            "invalid_request_error",
            code="method_not_allowed",
            extra_headers={"Allow": "POST"},
        )
    if path in GET_ONLY_V1_PATHS and req.method != "GET":
        return json_err(
            405,
            f"Method {req.method} not allowed for {path}. Use GET.",
            "invalid_request_error",
            code="method_not_allowed",
            extra_headers={"Allow": "GET"},
        )

    # 6b. Verify API Key / Session for /v1/* endpoints
    cred = extract_api_credential(req)
    if not cred:
        await asyncio.sleep(0.15)
        log_access(key=None, path=path, status=401, ip=ip)
        authdb.record_audit("anonymous", "api", f"unauthorized:{path}", 401, ip=ip)
        return json_err(
            401,
            "Missing or invalid API key. Pass `Authorization: Bearer sk-ornith-...` or `x-api-key: sk-ornith-...`.",
            "authentication_error",
            code="invalid_api_key",
        )

    key_name = cred["key_name"]
    user_email = cred["user_email"]
    train_ok = cred["train_ok"]
    allow_cmd = cred["allow_command"]
    rpm_limit = int(cred["rate_limit_rpm"] or 120)

    allowed, remaining = authdb.check_rate_limit(
        f"key:{cred['key_id']}:{user_email}", max_per_window=rpm_limit, window_secs=60.0
    )
    rl_headers = {
        "X-RateLimit-Limit-Requests": str(rpm_limit),
        "X-RateLimit-Remaining-Requests": str(remaining),
        "X-RateLimit-Reset-Requests": "60s",
    }
    if not allowed:
        return json_err(
            429,
            f"Rate limit exceeded ({rpm_limit} requests/min). Please retry after a few seconds.",
            "rate_limit_error",
            code="rate_limit_exceeded",
            extra_headers={"Retry-After": "10", **rl_headers},
        )

    # 7. Saved Chats & Feedback Endpoints
    if req.method == "GET" and path == "/v1/chats":
        rows = await loop.run_in_executor(
            None, chatdb.list_chats, key_name, req.query.get("limit", 50), req.query.get("before")
        )
        return web.json_response({"chats": rows}, headers=cors_headers(rl_headers))

    if req.method == "GET" and re.fullmatch(r"/v1/chats/[0-9a-f]{16}", path):
        c = await loop.run_in_executor(None, chatdb.get_chat, key_name, path.rsplit("/", 1)[1])
        return web.json_response(c, headers=cors_headers(rl_headers)) if c else json_err(404, "No such chat for this key.", extra_headers=rl_headers)

    if req.method == "GET" and re.fullmatch(r"/v1/images/[0-9a-f]{32}\.\w+", path):
        p = await loop.run_in_executor(None, chatdb.image_path, key_name, path.rsplit("/", 1)[1])
        return web.FileResponse(p, headers=cors_headers(rl_headers)) if p else json_err(404, "No such image for this key.", extra_headers=rl_headers)

    if req.method == "POST" and path == "/v1/feedback":
        try:
            fb = await req.json()
            assert isinstance(fb, dict) and fb.get("rating") in ("good", "bad") and re.fullmatch(r"[0-9a-f]{16}", fb.get("chat_id", ""))
        except Exception:
            return json_err(400, 'Send {"chat_id": "...", "rating": "good"|"bad", "correction": "..."}', extra_headers=rl_headers)
        ok = await loop.run_in_executor(
            None, chatdb.add_feedback, key_name, fb["chat_id"], fb["rating"], fb.get("correction"), fb.get("note")
        )
        return web.json_response({"ok": True}, headers=cors_headers(rl_headers)) if ok else json_err(404, "No such chat.", extra_headers=rl_headers)

    # 8. GET /v1/models (Deduplicated & Strictly Validated)
    if req.method == "GET" and path == "/v1/models":
        try:
            async with req.app["session"].get(f"{UPSTREAM}/v1/models", timeout=aiohttp.ClientTimeout(total=4)) as up:
                if up.status != 200:
                    return json_err(503, "Model inference service temporarily unavailable.", "service_unavailable", extra_headers={"Retry-After": "5", **rl_headers})
                raw_data = await up.json()
                status_by_id: Dict[str, str] = {}
                for m in raw_data.get("data", []):
                    if isinstance(m, dict) and m.get("id"):
                        status_by_id[m["id"]] = (m.get("status") or {}).get("value", "loaded")
                seen_ids = set()
                clean_models = []
                for card in MODEL_CATALOG:
                    mid = card["id"]
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    c_copy = dict(card)
                    if mid in status_by_id:
                        c_copy["status"] = status_by_id[mid]
                    clean_models.append(c_copy)
                return web.json_response({"object": "list", "data": clean_models}, headers=cors_headers(rl_headers))
        except Exception:
            return json_err(
                503,
                "Model inference service temporarily unavailable.",
                "service_unavailable",
                code="service_unavailable",
                extra_headers={"Retry-After": "5", **rl_headers},
            )

    # 8b. OpenAI-Compatible Image Generation & Editing Endpoints (/v1/images/generations, /v1/images/edits)
    if req.method == "POST" and path in ("/v1/images/generations", "/v1/images/edits"):
        try:
            body = await req.json()
        except Exception:
            return json_err(400, "JSON body required with `prompt`.", extra_headers=rl_headers)
        if not isinstance(body, dict):
            return json_err(400, "JSON body must be an object.", extra_headers=rl_headers)

        prompt = str(body.get("prompt") or body.get("instruction") or "").strip()
        if not prompt:
            return json_err(400, "`prompt` is required for image generation.", param="prompt", extra_headers=rl_headers)

        size_str = str(body.get("size") or "1024x1024").lower()
        width = int(body.get("width") or 1024)
        height = int(body.get("height") or 1024)
        if "x" in size_str:
            parts = size_str.split("x", 1)
            if parts[0].isdigit() and parts[1].isdigit():
                width, height = int(parts[0]), int(parts[1])

        count = max(1, min(4, int(body.get("n") or body.get("count") or 1)))
        resp_fmt = str(body.get("response_format") or "url").lower()
        include_b64 = resp_fmt in ("b64_json", "both", "base64")
        ref_img = body.get("image") if path == "/v1/images/edits" else None

        t0 = time.time()
        res = await server_tools.generate_images_direct(
            prompt=prompt[:2000],
            width=width,
            height=height,
            count=count,
            ref_image_b64=ref_img,
            include_b64=include_b64,
        )
        latency_ms = (time.time() - t0) * 1000
        if "error" in res:
            authdb.record_audit(user_email, "image", path, 503, latency_ms=latency_ms, ip=ip, detail=res)
            return json_err(503, res["error"], "server_error", extra_headers={"Retry-After": "5", **rl_headers})

        authdb.record_key_usage(cred["key_id"], 0, 0)
        authdb.record_audit(
            user_email,
            "image",
            f"{path} ({width}x{height} x{count})",
            200,
            latency_ms=latency_ms,
            ip=ip,
            detail={"prompt": prompt[:160], "urls": [d["url"] for d in res.get("data", [])]},
        )
        return web.json_response(res, headers=cors_headers(rl_headers))

    # 9. Model Inference Endpoints (/v1/chat/completions, /v1/completions, /v1/messages, /v1/messages/count_tokens, /v1/responses)
    if req.method == "POST" and path in (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/responses",
    ):
        if (req.content_length or 0) > MAX_BODY:
            return json_err(413, "Request payload exceeds 35 MB limit.", extra_headers=rl_headers)
        try:
            body = await req.json()
        except Exception:
            return json_err(400, "Request body must be valid JSON.", code="invalid_json", extra_headers=rl_headers)

        if not isinstance(body, dict):
            return json_err(400, "Request body must be a JSON object.", code="invalid_json", extra_headers=rl_headers)

        # Validate model name first (returns 404 model_not_found on unknown models; defaults to ornith-1.5-35b-uncensored if omitted)
        client_model, upstream_model, model_err = resolve_and_validate_model(body.get("model"))
        if model_err or not client_model or not upstream_model:
            return json_err(
                404,
                model_err or "Model not found.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
                extra_headers=rl_headers,
            )

        # Validate request parameters (messages, roles, max_tokens, temperature, top_p, n, response_format)
        val_err = validate_chat_or_completion_input(path, body)
        if val_err:
            st_code, err_msg, err_type, err_param, err_code = val_err
            return json_err(st_code, err_msg, err_type, param=err_param, code=err_code, extra_headers=rl_headers)

        body["model"] = upstream_model
        t0 = time.time()

        guardrail_block = _check_inference_host_guardrail(body)
        if guardrail_block:
            authdb.record_audit(user_email, "security", f"blocked_host_exfil:{path}", 403, ip=ip)
            return json_err(403, guardrail_block, "security_guardrail_error", code="host_guardrail_blocked", extra_headers=rl_headers)

        # Configure thinking mode:
        # Default enable_thinking=False so visible content streams immediately and small max_tokens (e.g. 16)
        # never exhaust their budget on hidden reasoning tokens.
        explicit_think = body.pop("enable_thinking", None)
        ct_kwargs = dict(body.get("chat_template_kwargs") or {})
        max_tok_val = body.get("max_tokens") or body.get("max_completion_tokens")
        rf_type = (body.get("response_format") or {}).get("type") if isinstance(body.get("response_format"), dict) else None

        if rf_type in ("json_object", "json_schema") or (isinstance(max_tok_val, int) and max_tok_val <= 64):
            # Force thinking off for JSON grammar mode or tiny max_tokens so content is never empty
            ct_kwargs["enable_thinking"] = False
        elif explicit_think is not None:
            ct_kwargs["enable_thinking"] = bool(explicit_think)
        elif "enable_thinking" not in ct_kwargs:
            has_reasoning_opt = bool(body.get("reasoning_effort")) or bool(body.get("thinking"))
            ct_kwargs["enable_thinking"] = has_reasoning_opt
        body["chat_template_kwargs"] = ct_kwargs

        # Enforce greedy determinism when temperature == 0 or seed is provided
        if body.get("temperature") is not None and float(body["temperature"]) == 0.0:
            body["temperature"] = 0.0
            body["top_k"] = 1
            body["top_p"] = 1.0
            if body.get("seed") is None:
                body["seed"] = 1
        elif body.get("seed") is not None:
            try:
                body["seed"] = int(body["seed"])
            except Exception:
                body["seed"] = 1

        # 9a. Anthropic /v1/messages & /v1/messages/count_tokens (Claude Code CLI & Anthropic SDK)
        if path in ("/v1/messages", "/v1/messages/count_tokens"):
            sys_parts: List[str] = []
            raw_sys = body.get("system")
            if isinstance(raw_sys, str) and raw_sys.strip():
                sys_parts.append(raw_sys.strip())
            elif isinstance(raw_sys, list):
                for item in raw_sys:
                    if isinstance(item, str) and item.strip():
                        sys_parts.append(item.strip())
                    elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                        sys_parts.append(item["text"].strip())

            if isinstance(body.get("messages"), list):
                cleaned_msgs = []
                for m in body["messages"]:
                    if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                        mc = m.get("content")
                        if isinstance(mc, str) and mc.strip():
                            sys_parts.append(mc.strip())
                        elif isinstance(mc, list):
                            for part in mc:
                                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                                    sys_parts.append(part["text"].strip())
                    else:
                        cleaned_msgs.append(m)
                body["messages"] = cleaned_msgs

            if not cleaned_msgs:
                return json_err(400, "'messages' must contain at least one user or assistant message.", param="messages", extra_headers=rl_headers)

            if sys_parts:
                body["system"] = "\n\n".join(sys_parts)
            else:
                body["system"] = MODEL_IDENTITY_NOTES.get(client_model, MODEL_IDENTITY_NOTES[PRIMARY_MODEL])

            stream = bool(body.get("stream")) and path == "/v1/messages"
            try:
                up = await req.app["session"].post(
                    UPSTREAM + path,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None),
                )
            except Exception:
                return json_err(503, "Model inference server is temporarily unreachable.", "service_unavailable", extra_headers={"Retry-After": "5", **rl_headers})

            hdrs = {"X-Ornith-Model": client_model, **cors_headers(rl_headers)}
            try:
                if up.status != 200:
                    raw_err = await up.read()
                    st_err, err_payload = parse_upstream_error(up.status, raw_err)
                    return web.json_response(err_payload, status=st_err, headers=hdrs)

                if stream:
                    resp = web.StreamResponse(
                        status=200,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                            **hdrs,
                        },
                    )
                    await resp.prepare(req)
                    await resp.write(b": stream-open\n\n")
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    latency_ms = (time.time() - t0) * 1000
                    authdb.record_key_usage(cred["key_id"], 0, 0)
                    authdb.record_audit(user_email, "anthropic", f"{path} (stream)", 200, latency_ms=latency_ms, ip=ip)
                    return resp
                else:
                    data = await up.read()
                    latency_ms = (time.time() - t0) * 1000
                    tin, tout = 0, 0
                    try:
                        parsed = json.loads(data)
                        u = parsed.get("usage") or {}
                        tin = int(u.get("input_tokens") or 0)
                        tout = int(u.get("output_tokens") or 0)
                    except Exception:
                        pass
                    authdb.record_key_usage(cred["key_id"], tin, tout)
                    authdb.record_audit(user_email, "anthropic", path, 200, latency_ms=latency_ms, tokens_in=tin, tokens_out=tout, ip=ip)
                    return web.Response(body=data, status=200, content_type="application/json", headers=hdrs)
            finally:
                up.release()

        # 9b. OpenAI Codex /v1/responses
        if path == "/v1/responses":
            body, _, _ = _normalize_responses_payload(body, upstream_model)
            stream = bool(body.get("stream", True))
            try:
                up = await req.app["session"].post(
                    f"{UPSTREAM}/v1/responses",
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None),
                )
            except Exception:
                return json_err(503, "Model inference server is temporarily unreachable.", "service_unavailable", extra_headers={"Retry-After": "5", **rl_headers})

            hdrs = {"X-Ornith-Model": client_model, **cors_headers(rl_headers)}
            try:
                if up.status != 200:
                    raw_err = await up.read()
                    st_err, err_payload = parse_upstream_error(up.status, raw_err)
                    return web.json_response(err_payload, status=st_err, headers=hdrs)

                if stream:
                    resp = web.StreamResponse(
                        status=200,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                            **hdrs,
                        },
                    )
                    await resp.prepare(req)
                    await resp.write(b": stream-open\n\n")
                    async for chunk in up.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    latency_ms = (time.time() - t0) * 1000
                    authdb.record_key_usage(cred["key_id"], 0, 0)
                    authdb.record_audit(user_email, "responses", "/v1/responses (stream)", 200, latency_ms=latency_ms, ip=ip)
                    return resp
                else:
                    data = await up.read()
                    latency_ms = (time.time() - t0) * 1000
                    authdb.record_key_usage(cred["key_id"], 0, 0)
                    authdb.record_audit(user_email, "responses", "/v1/responses", 200, latency_ms=latency_ms, ip=ip)
                    return web.Response(body=data, status=200, content_type="application/json", headers=hdrs)
            finally:
                up.release()

        # 9c. OpenAI /v1/chat/completions & /v1/completions
        if path == "/v1/chat/completions":
            sys_chunks: List[str] = []
            non_sys_msgs: List[Dict[str, Any]] = []
            for m in body["messages"]:
                if m.get("role") in ("system", "developer"):
                    mc = m.get("content")
                    if isinstance(mc, str) and mc.strip():
                        sys_chunks.append(mc.strip())
                    elif isinstance(mc, list):
                        for part in mc:
                            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                                sys_chunks.append(part["text"].strip())
                else:
                    non_sys_msgs.append(m)

            if not non_sys_msgs:
                return json_err(
                    400,
                    "'messages' must include at least one non-system message (e.g. role 'user').",
                    "invalid_request_error",
                    param="messages",
                    code="invalid_messages",
                    extra_headers=rl_headers,
                )

            # Use a compact model-identity system note ONLY when no system prompt was provided (or for ornith-calendar)
            # This adds only ~10 tokens (eliminating the old 1,045-token overhead) while ensuring each model
            # accurately knows its Ornith identity.
            if not sys_chunks:
                sys_chunks.append(MODEL_IDENTITY_NOTES.get(client_model, MODEL_IDENTITY_NOTES[PRIMARY_MODEL]))
            elif client_model == "ornith-calendar" and "ornith-calendar" not in " ".join(sys_chunks).lower():
                sys_chunks.insert(0, MODEL_IDENTITY_NOTES["ornith-calendar"])

            body["messages"] = [{"role": "system", "content": "\n\n".join(sys_chunks)}, *non_sys_msgs]

            # llama-server grammar fails if both `tools` and `response_format: json_object` are passed together
            if rf_type in ("json_object", "json_schema") and body.get("tools"):
                body.pop("tools", None)
                body.pop("tool_choice", None)

        stream = bool(body.get("stream"))
        chat_id = uuid.uuid4().hex[:16]
        conv = chatdb.conversation_of(key_name, body, req.headers.get("X-Conversation-Id"))

        def save_turn(status, answer, reasoning, tools, usage):
            secs = round(time.time() - t0, 1)
            tin = int((usage or {}).get("prompt_tokens") or 0)
            tout = int((usage or {}).get("completion_tokens") or 0)
            log_access(
                key=key_name,
                user=user_email,
                path=path,
                status=status,
                secs=secs,
                chat=chat_id,
                stream=stream,
                model=client_model,
                tokens_in=tin,
                tokens_out=tout,
            )
            authdb.record_key_usage(cred["key_id"], tin, tout)
            authdb.record_audit(
                user_email,
                "chat",
                f"{path} ({client_model})",
                status,
                latency_ms=secs * 1000,
                tokens_in=tin,
                tokens_out=tout,
                ip=ip,
                detail={"chat_id": chat_id, "tools_used": [t.get("tool") for t in (tools or [])]},
            )
            loop.run_in_executor(
                None,
                chatdb.save,
                chat_id,
                key_name,
                train_ok,
                path,
                body,
                conv,
                status,
                secs,
                stream,
                answer,
                reasoning,
                tools,
                usage,
            )

        # Server tools (sandbox, web_search, generate_image) are STRICTLY OPT-IN:
        # Only activated when caller explicitly passes `server_tools: true`, non-empty `enabled_tools`,
        # or `model: "ornith-1.5-35b-agent"`.
        enabled_tool_names = body.pop("enabled_tools", None)
        server_tools_flag = body.pop("server_tools", None)
        should_use_server_tools = (
            (
                server_tools_flag is True
                or (isinstance(enabled_tool_names, list) and len(enabled_tool_names) > 0)
                or client_model == "ornith-1.5-35b-agent"
            )
            and rf_type not in ("json_object", "json_schema")
            and not body.get("tools")
            and cred["allow_tools"]
        )

        if path == "/v1/chat/completions" and should_use_server_tools:
            async with _LONG_REQ_SEM:
                return await run_agent_with_server_tools(
                    req,
                    body,
                    chat_id,
                    save_turn,
                    UPSTREAM,
                    client_model,
                    upstream_model,
                    allow_command=allow_cmd,
                    enabled_tool_names=enabled_tool_names,
                    rl_headers=rl_headers,
                )

        hdrs = {"X-Chat-Id": chat_id, "X-Ornith-Model": client_model, **cors_headers(rl_headers)}

        # Handle n > 1 cleanly for non-streaming /v1/chat/completions so it never deadlocks the 2-slot GPU router
        n_choices = int(body.get("n") or 1)
        if path == "/v1/chat/completions" and n_choices > 1 and not stream:
            single_body = {**body, "n": 1, "stream": False}
            choices_out = []
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            first_meta: Dict[str, Any] = {}
            for choice_idx in range(n_choices):
                req_copy = dict(single_body)
                if req_copy.get("seed") is not None and float(req_copy.get("temperature", 0.7)) > 0:
                    req_copy["seed"] = int(req_copy["seed"]) + choice_idx
                try:
                    up = await req.app["session"].post(
                        UPSTREAM + path,
                        json=req_copy,
                        timeout=aiohttp.ClientTimeout(total=120, connect=10),
                    )
                except Exception:
                    return json_err(503, "Model inference server is temporarily unreachable.", "service_unavailable", extra_headers={"Retry-After": "5", **rl_headers})
                raw_b = await up.read()
                up.release()
                if up.status != 200:
                    st_err, err_payload = parse_upstream_error(up.status, raw_b)
                    return web.json_response(err_payload, status=st_err, headers=hdrs)
                parsed = json.loads(raw_b)
                if not first_meta:
                    first_meta = parsed
                ch = (parsed.get("choices") or [{}])[0]
                ch["index"] = choice_idx
                choices_out.append(ch)
                u = parsed.get("usage") or {}
                if choice_idx == 0:
                    total_usage["prompt_tokens"] = int(u.get("prompt_tokens") or 0)
                total_usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
            total_usage["total_tokens"] = total_usage["prompt_tokens"] + total_usage["completion_tokens"]
            combined_resp = {
                "id": first_meta.get("id") or f"chatcmpl-{chat_id}",
                "object": "chat.completion",
                "created": first_meta.get("created") or int(time.time()),
                "model": client_model,
                "choices": choices_out,
                "usage": total_usage,
            }
            ans0 = ((choices_out[0].get("message") or {}).get("content")) if choices_out else ""
            save_turn(200, ans0, None, None, total_usage)
            return web.json_response(combined_resp, status=200, headers=hdrs)

        # For /v1/chat/completions, ALWAYS stream from llama-server upstream (even when client stream=False!)
        # Why?
        # 1. Upstream errors (like 161K token context overflow) return HTTP 400 in <0.4s before any 200 header is sent.
        # 2. Long non-streaming requests (>100s) receive HTTP 200 + periodic whitespace keepalives (`b" "`) after 6s,
        #    completely defeating Cloudflare's 100-second non-streaming origin timeout (524)!
        # 3. Long streaming requests receive `: keepalive\n\n` comments and a guaranteed closing `finish_reason` + `[DONE]`.
        use_long_sem = is_long_request(body, server_tools_enabled=False)

        async def _execute_chat_or_completion() -> web.StreamResponse:
            if path == "/v1/chat/completions":
                want_usage = bool((body.get("stream_options") or {}).get("include_usage", True))
                up_body = {
                    **body,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                try:
                    up = await req.app["session"].post(
                        UPSTREAM + path,
                        json=up_body,
                        timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None),
                    )
                except Exception:
                    return json_err(
                        503,
                        "Model inference server is temporarily unreachable.",
                        "service_unavailable",
                        code="service_unavailable",
                        extra_headers={"Retry-After": "5", **rl_headers},
                    )

                try:
                    # Check upstream status BEFORE preparing a 200 response!
                    if up.status != 200:
                        raw_err = await up.read()
                        st_err, err_payload = parse_upstream_error(up.status, raw_err)
                        save_turn(st_err, None, None, None, {})
                        return web.json_response(err_payload, status=st_err, headers=hdrs)

                    col = chatdb.StreamCollector()

                    if stream:
                        resp = web.StreamResponse(
                            status=200,
                            headers={
                                "Content-Type": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                                **hdrs,
                            },
                        )
                        await resp.prepare(req)
                        await resp.write(b": stream-open\n\n")

                        it = up.content.iter_any().__aiter__()
                        stream_aborted = False
                        while True:
                            try:
                                chunk = await asyncio.wait_for(it.__anext__(), timeout=8.0)
                            except asyncio.TimeoutError:
                                await resp.write(b": keepalive\n\n")
                                continue
                            except StopAsyncIteration:
                                break
                            except Exception:
                                stream_aborted = True
                                break

                            col.feed(chunk)
                            # Rewrite model name in chunk if client requested an alias like ornith-1.5-35b
                            if client_model != upstream_model and b'"model":' in chunk:
                                chunk = chunk.replace(
                                    f'"model":"{upstream_model}"'.encode(),
                                    f'"model":"{client_model}"'.encode(),
                                )
                            await resp.write(chunk)

                        # Guarantee a closing finish_reason chunk and data: [DONE] even if upstream closed early
                        if not col.finish_reason:
                            fallback_fin = "length" if (
                                isinstance(max_tok_val, int)
                                and (col.usage.get("completion_tokens") or 0) >= max_tok_val
                            ) else "stop"
                            closing_ev = {
                                "id": col.cmpl_id or f"chatcmpl-{chat_id}",
                                "object": "chat.completion.chunk",
                                "created": col.created or int(time.time()),
                                "model": client_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": fallback_fin}],
                            }
                            await resp.write(b"data: " + json.dumps(closing_ev).encode("utf-8") + b"\n\n")
                        if not col.saw_done:
                            await resp.write(b"data: [DONE]\n\n")

                        await resp.write_eof()
                        save_turn(200 if not stream_aborted else 500, *col.result())
                        return resp

                    else:
                        # Client requested stream=False, but we consume SSE from llama-server so we can:
                        # (a) return immediately if finished within 6 seconds, OR
                        # (b) stream whitespace keepalives (`b" "`) every 5s so Cloudflare 100s timeout never triggers!
                        async def _consume_all() -> None:
                            async for chunk in up.content.iter_any():
                                col.feed(chunk)

                        consume_task = asyncio.ensure_future(_consume_all())
                        done, _ = await asyncio.wait({consume_task}, timeout=6.0)
                        if consume_task in done:
                            consume_task.result()
                            payload_out = col.build_chat_completion(f"chatcmpl-{chat_id}", client_model)
                            save_turn(200, *col.result())
                            return web.json_response(payload_out, status=200, headers=hdrs)

                        # Long non-streaming generation (>6s): open 200 application/json stream and send b" " every 5s
                        resp = web.StreamResponse(
                            status=200,
                            headers={
                                "Content-Type": "application/json",
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                                **hdrs,
                            },
                        )
                        await resp.prepare(req)
                        await resp.write(b" ")
                        while not consume_task.done():
                            await asyncio.wait({consume_task}, timeout=5.0)
                            if not consume_task.done():
                                await resp.write(b" ")
                        consume_task.result()
                        payload_out = col.build_chat_completion(f"chatcmpl-{chat_id}", client_model)
                        await resp.write(json.dumps(payload_out).encode("utf-8"))
                        await resp.write_eof()
                        save_turn(200, *col.result())
                        return resp
                finally:
                    up.release()

            # Legacy /v1/completions path
            try:
                up = await req.app["session"].post(
                    UPSTREAM + path,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None),
                )
            except Exception:
                return json_err(
                    503,
                    "Model inference server is temporarily unreachable.",
                    "service_unavailable",
                    extra_headers={"Retry-After": "5", **rl_headers},
                )
            try:
                if up.status != 200:
                    raw_err = await up.read()
                    st_err, err_payload = parse_upstream_error(up.status, raw_err)
                    return web.json_response(err_payload, status=st_err, headers=hdrs)
                data = await up.read()
                save_turn(200, *chatdb.from_response(data))
                return web.Response(body=data, status=200, content_type="application/json", headers=hdrs)
            finally:
                up.release()

        if use_long_sem:
            async with _LONG_REQ_SEM:
                return await _execute_chat_or_completion()
        else:
            return await _execute_chat_or_completion()

    return json_err(404, f"Endpoint {path} not found. Visit / for the Ornith Developer Console & API Docs.")


async def run_agent_with_server_tools(
    req: web.Request,
    body: Dict[str, Any],
    chat_id: str,
    save_turn,
    target_upstream: str = UPSTREAM,
    client_model: str = PRIMARY_MODEL,
    upstream_model: str = PRIMARY_MODEL,
    allow_command: bool = True,
    enabled_tool_names: Optional[List[str]] = None,
    rl_headers: Optional[Dict[str, str]] = None,
) -> web.StreamResponse:
    """Opt-in multi-round server-tool loop (web_search, read_webpage, generate_image, edit_image, run_sandbox_code, run_command).
    Capped by steps (max 3 tool rounds) and time (45s budget), never returns HTTP 200 on Round-0 upstream errors,
    and ensures any sandbox code executed is included in `content` if the model only narrated the result.
    """
    stream = bool(body.get("stream"))
    want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    msgs = [dict(m) for m in body.get("messages", [])]
    note = server_tools.system_note(allow_command=allow_command)
    if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
        msgs[0]["content"] += "\n\n" + note
    else:
        msgs.insert(0, {"role": "system", "content": note})

    active_tools = server_tools.get_tools_for_caller(allow_command=allow_command, enabled_names=enabled_tool_names)
    base = {
        k: v
        for k, v in body.items()
        if k not in ("messages", "stream", "stream_options", "tools", "tool_choice", "n")
    }
    base["model"] = upstream_model
    cid, created = "chatcmpl-" + chat_id, int(time.time())
    hdrs = {
        "X-Chat-Id": chat_id,
        "X-Ornith-Model": client_model,
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        **cors_headers(rl_headers),
    }

    resp: Optional[web.StreamResponse] = None

    async def ensure_prepared() -> web.StreamResponse:
        nonlocal resp
        if resp is None:
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream" if stream else "application/json", **hdrs},
            )
            await resp.prepare(req)
            if stream:
                await resp.write(b": stream-open\n\n")
        return resp

    async def sse(delta, finish=None, usage=None, tool_trace=None):
        r = await ensure_prepared()
        ev: Dict[str, Any] = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": client_model,
            "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            ev["usage"] = usage
        if tool_trace:
            ev["ornith_tool_event"] = tool_trace
        await r.write(b"data: " + json.dumps(ev).encode() + b"\n\n")

    async def keepalive(coro):
        r = await ensure_prepared()
        task = asyncio.ensure_future(coro)
        while True:
            done, _ = await asyncio.wait({task}, timeout=6)
            if done:
                return task.result()
            await r.write(b": working\n\n" if stream else b" ")

    content, reasoning, trace, generated, sandbox_snippets = [], [], [], [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status = 200
    last_finish_reason = "stop"
    nudged = False
    loop_start = time.time()

    async def emit(line: str):
        reasoning.append(line)
        if stream:
            await sse({"reasoning_content": "\n" + line + "\n"})

    max_rounds = 4
    for rnd in range(max_rounds):
        elapsed = time.time() - loop_start
        allow_tools_this_round = (rnd < max_rounds - 1) and bool(active_tools) and (elapsed < 40.0)
        payload = {**base, "messages": msgs, "stream": True, "stream_options": {"include_usage": True}}
        if allow_tools_this_round:
            payload["tools"] = active_tools

        try:
            up = await req.app["session"].post(
                target_upstream + "/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None),
            )
        except Exception:
            if resp is None:
                return json_err(
                    503,
                    "Model inference server is temporarily unreachable.",
                    "service_unavailable",
                    extra_headers={"Retry-After": "5", **(rl_headers or {})},
                )
            break

        if up.status != 200:
            raw_err = await up.read()
            up.release()
            st_err, err_payload = parse_upstream_error(up.status, raw_err)
            if resp is None:
                # Never return HTTP 200 on Round 0 error! Return actual HTTP status code.
                save_turn(st_err, None, None, None, {})
                return web.json_response(err_payload, status=st_err, headers=hdrs)
            status = st_err
            break

        await ensure_prepared()
        col = chatdb.StreamCollector()
        buf = b""

        async def pump():
            nonlocal buf
            async for chunk in up.content.iter_any():
                col.feed(chunk)
                buf += chunk
                if stream:
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                            continue
                        try:
                            ev = json.loads(line[5:])
                        except Exception:
                            continue
                        for ch in ev.get("choices") or []:
                            d = ch.get("delta") or {}
                            out = {k: d[k] for k in ("content", "reasoning_content") if d.get(k)}
                            if out:
                                await sse(out)

        try:
            await pump() if stream else await keepalive(pump())
        finally:
            up.release()

        text, think, calls, u = col.result()
        if col.finish_reason in ("stop", "length"):
            last_finish_reason = col.finish_reason
        for k in usage:
            usage[k] += (u or {}).get(k) or 0
        if text:
            content.append(text)
        if think:
            reasoning.append(think)
        if not calls:
            real = [g for g in generated if g in (text or "")]
            if (
                "![" in (text or "")
                and not real
                and not nudged
                and rnd < max_rounds - 1
                and any(t["function"]["name"] == "generate_image" for t in active_tools)
            ):
                nudged = True
                if content and content[-1] == text:
                    content.pop()
                msgs.append({"role": "assistant", "content": text})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "[system] That image URL does not exist because `generate_image` was not called. "
                            "Call `generate_image` now to create the image, then include its returned URL."
                        ),
                    }
                )
                if stream:
                    await sse({"content": "\n\n"})
                continue
            break

        msgs.append({"role": "assistant", "content": text or "", "tool_calls": calls})
        for c in calls:
            fn = c.get("function") or {}
            t_name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if t_name == "run_sandbox_code" and isinstance(args.get("code"), str) and args["code"].strip():
                lang = str(args.get("language") or "python")
                sandbox_snippets.append((lang, args["code"].strip()))
            if stream:
                await sse(
                    {"reasoning_content": f"\n[executing {t_name}: {json.dumps(args)[:180]}]\n"},
                    tool_trace={"phase": "start", "tool": t_name, "args": args},
                )
            result, line = await keepalive(
                server_tools.run(t_name, args, body.get("messages", []), generated, emit, allow_command=allow_command)
            )
            await emit(line)
            if stream:
                await sse(
                    {},
                    tool_trace={"phase": "done", "tool": t_name, "args": args, "result": result[:1200]},
                )
            msgs.append({"role": "tool", "tool_call_id": c.get("id") or t_name, "content": result[:14000]})
            trace.append({"tool": t_name, "args": args, "result": result[:1200]})

    r = await ensure_prepared()
    answer = "".join(content).strip()

    # If the model executed code in `run_sandbox_code` and only narrated the outcome in `content` without
    # including a code block, append the verified sandbox code block into `content`!
    if sandbox_snippets and "```" not in answer:
        last_lang, last_code = sandbox_snippets[-1]
        code_block = f"\n\n```{last_lang}\n{last_code}\n```"
        answer = (answer + code_block).strip()
        if stream:
            await sse({"content": code_block})

    images = [f"{server_tools.PUBLIC_BASE}/files/{g}" for g in generated]
    missing_md = [
        f"![Generated Image]({server_tools.PUBLIC_BASE}/files/{g})"
        for g in generated
        if g not in answer
    ]
    if missing_md:
        extra_md = ("\n\n" if answer else "") + "\n\n".join(missing_md)
        answer = (answer + extra_md).strip()
        if stream:
            await sse({"content": extra_md, "images": images})
    if stream:
        await sse({}, finish=last_finish_reason)
        if want_usage:
            await sse({}, usage=usage)
        await r.write(b"data: [DONE]\n\n")
    else:
        await r.write(
            json.dumps(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": client_model,
                    "usage": usage,
                    "images": images,
                    "tool_trace": trace,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": last_finish_reason,
                            "message": {
                                "role": "assistant",
                                "content": answer,
                                "images": images,
                                "reasoning_content": "\n".join(reasoning) or None,
                            },
                        }
                    ],
                }
            ).encode("utf-8")
        )
    await r.write_eof()
    save_turn(status, answer, "\n".join(reasoning) or None, trace or None, usage)
    return r


async def app_ctx(app: web.Application):
    app["session"] = aiohttp.ClientSession()
    watchdog_task = asyncio.create_task(always_on_watchdog(app))
    yield
    watchdog_task.cancel()
    await app["session"].close()


def main() -> None:
    authdb.init_db()
    chatdb.connect().close()
    app = web.Application(client_max_size=MAX_BODY)
    app.cleanup_ctx.append(app_ctx)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host="127.0.0.1", port=8099, access_log=None)


if __name__ == "__main__":
    main()
