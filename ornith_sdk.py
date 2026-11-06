#!/usr/bin/env python3
"""
Ornith-1.5-35B-Uncensored Python SDK, Authenticator & Coding Agent CLI (`ornith`)

Install anywhere:
  curl -fsSL https://kusi-qwen-api.vercel.app/install.sh | bash

Authenticate (choose Developer Account via Resend 6-digit code OR API Key):
  ornith login
  ornith login --account 2024frank@gmail.com
  ornith login --api-key sk-ornith-...
  ornith whoami
  ornith logout

Interactive Coding Agent & One-Shot Prompts:
  ornith code
  ornith code "Inspect this directory and write unit tests"
  ornith "Write a lock-free ring buffer in C++20"
  ornith --image ui.png "Critique this screenshot"
  ornith --video clip.mp4 "Summarize this video clip"
  ornith sandbox --lang python "import math; print(math.factorial(20))"
  ornith status
"""

import argparse
import base64
import getpass
import glob
import json
import mimetypes
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

DEFAULT_PUBLIC_URL = "https://kusi-qwen-api.vercel.app"
_LOCAL_GATEWAY = "http://127.0.0.1:8099"
DEFAULT_MODEL = "ornith-1.5-35b-uncensored"
CONFIG_DIR = pathlib.Path.home() / ".config" / "ornith"
CONFIG_FILE = CONFIG_DIR / "config.json"
KEY_FILE = CONFIG_DIR / "api_key"
ENV_FILE = CONFIG_DIR / "env.sh"


def _resolve_default_url() -> str:
    if os.environ.get("ORNITH_BASE_URL"):
        return os.environ["ORNITH_BASE_URL"].strip().rstrip("/")
    cfg = load_config()
    if cfg.get("base_url"):
        return str(cfg["base_url"]).rstrip("/")
    # Probe local gateway quietly first; fall back to public endpoint
    try:
        req = urllib.request.Request(f"{_LOCAL_GATEWAY}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            if resp.status == 200:
                return _LOCAL_GATEWAY
    except Exception:
        pass
    return DEFAULT_PUBLIC_URL


def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(
    api_key: str,
    base_url: str,
    auth_method: str = "api_key",
    email: str = "",
    name: str = "",
    role: str = "developer",
    session_token: str = "",
) -> None:
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    cfg = {
        "api_key": api_key.strip(),
        "base_url": base_url.rstrip("/"),
        "auth_method": auth_method,
        "email": email,
        "name": name,
        "role": role,
        "session_token": session_token,
        "model": DEFAULT_MODEL,
    }
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.chmod(CONFIG_FILE, 0o600)

    if api_key.strip():
        KEY_FILE.write_text(api_key.strip() + "\n", encoding="utf-8")
        os.chmod(KEY_FILE, 0o600)
        env_text = (
            f'# Ornith Platform Studio CLI Environment\n'
            f'export ORNITH_API_KEY="{api_key.strip()}"\n'
            f'export ORNITH_BASE_URL="{base_url.rstrip("/")}"\n'
            f'export ORNITH_MODEL="{DEFAULT_MODEL}"\n'
            f'export OPENAI_API_KEY="{api_key.strip()}"\n'
            f'export OPENAI_BASE_URL="{base_url.rstrip("/")}/v1"\n'
            f'export ANTHROPIC_API_KEY="{api_key.strip()}"\n'
            f'export ANTHROPIC_AUTH_TOKEN="{api_key.strip()}"\n'
            f'export ANTHROPIC_BASE_URL="{base_url.rstrip("/")}"\n'
        )
        ENV_FILE.write_text(env_text, encoding="utf-8")
        os.chmod(ENV_FILE, 0o600)


def load_api_key() -> str:
    if os.environ.get("ORNITH_API_KEY"):
        return os.environ["ORNITH_API_KEY"].strip()
    cfg = load_config()
    if cfg.get("api_key"):
        return str(cfg["api_key"]).strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


class OrnithClient:
    """Universal Python client for Ornith-1.5-35B-Uncensored (OpenAI, Anthropic, Sandbox & Tools)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.base_url = (base_url or _resolve_default_url()).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.api_key = api_key if api_key is not None else load_api_key()
        self.model = model

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "User-Agent": "OrnithCLI/2.5",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _encode_image(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://", "data:")):
            return path_or_url
        p = pathlib.Path(path_or_url).expanduser()
        mime, _ = mimetypes.guess_type(str(p))
        mime = mime or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def extract_video_frames(self, video_path: str, max_frames: int = 4) -> List[str]:
        p = pathlib.Path(video_path).expanduser()
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        req = urllib.request.Request(
            f"{self.base_url}/api/media/extract-frames",
            data=json.dumps({"video_base64": b64, "ext": p.suffix or ".mp4", "max_frames": max_frames}).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("frames", [])

    def chat(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        video: Optional[str] = None,
        enable_thinking: bool = True,
        server_tools: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Union[str, Dict[str, Any], Generator[Dict[str, str], None, None]]:
        if isinstance(prompt, list):
            messages = list(prompt)
        else:
            content: Any = prompt
            media_urls: List[str] = []
            for img in images or []:
                media_urls.append(self._encode_image(img))
            if video:
                media_urls.extend(self.extract_video_frames(video))
            if media_urls:
                content = [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": u}} for u in media_urls],
                ]
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "server_tools": server_tools and not tools,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if tools:
            payload["tools"] = tools

        if stream:
            return self._stream_chat(payload)

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if tools:
                return data["choices"][0]["message"]
            return data["choices"][0]["message"]["content"]

    def _stream_chat(self, payload: Dict[str, Any]) -> Generator[Dict[str, str], None, None]:
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_part = line[5:].strip()
                if not data_part or data_part == "[DONE]":
                    continue
                try:
                    ev = json.loads(data_part)
                    delta = ((ev.get("choices") or [{}])[0]).get("delta") or {}
                    if delta.get("reasoning_content") or delta.get("content"):
                        yield {
                            "reasoning": delta.get("reasoning_content") or "",
                            "content": delta.get("content") or "",
                        }
                except Exception:
                    continue

    def run_sandbox(self, code: str, language: str = "python", timeout: int = 25) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/api/sandbox/run",
            data=json.dumps({"mode": "sandbox", "language": language, "code": code, "timeout": timeout}).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def run_command(self, command: str, cwd: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/api/sandbox/run",
            data=json.dumps({"mode": "command", "command": command, "cwd": cwd, "timeout": timeout}).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def status(self) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/api/hardware", headers=self._headers())
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def me(self) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/api/auth/me", headers=self._headers())
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate_image(self, prompt: str, size: str = "1024x1024", include_b64: bool = False) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/v1/images/generations",
            data=json.dumps({
                "prompt": prompt,
                "size": size,
                "response_format": "both" if include_b64 else "url",
            }).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ── Authentication Flows (Account OTP via Resend OR API Key) ─────────────

def _display_endpoint(url: str) -> str:
    if "127.0.0.1" in url or "localhost" in url:
        return "https://kusi-qwen-api.vercel.app (Local Direct)"
    return url


def handle_login(client: OrnithClient, account_email: Optional[str], given_api_key: Optional[str], otp_code: Optional[str]) -> int:
    print("\033[1mOrnith Platform Studio — CLI Authentication\033[0m")
    print(f"Gateway: {_display_endpoint(client.base_url)}\n")

    choice = ""
    if given_api_key:
        choice = "2"
    elif account_email:
        choice = "1"
    else:
        print("Choose how you want to authenticate:")
        print("  \033[1m[1]\033[0m Developer Account (Email + 6-digit Resend verification code)")
        print("  \033[1m[2]\033[0m API Key (\033[90msk-ornith-...\033[0m)")
        try:
            choice = input("\nSelect option [1/2] (default: 1): ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 1

    if choice == "1":
        email = (account_email or "").strip()
        if not email:
            try:
                email = input("Developer Email Address: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 1
        if not email:
            print("Error: Email address is required.", file=sys.stderr)
            return 1

        if not otp_code:
            print(f"Sending 6-digit verification code to {email} via Resend...")
            req = urllib.request.Request(
                f"{client.base_url}/api/auth/request-code",
                data=json.dumps({"email": email}).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "OrnithCLI/2.5"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    print(f"✓ {res_data.get('message', 'Code sent.')}")
            except urllib.error.HTTPError as he:
                err_txt = he.read().decode("utf-8", errors="ignore")
                try:
                    msg = json.loads(err_txt).get("error", {}).get("message", err_txt)
                except Exception:
                    msg = err_txt
                print(f"✗ Failed to request code: {msg}", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"✗ Connection error: {exc}", file=sys.stderr)
                return 1

            try:
                otp_code = input("Enter 6-digit code from your email: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 1

        vreq = urllib.request.Request(
            f"{client.base_url}/api/auth/verify-code",
            data=json.dumps({
                "email": email,
                "code": otp_code,
                "mint_cli_key": True,
                "client_name": "ornith-cli",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "OrnithCLI/2.5"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(vreq, timeout=15) as resp:
                vdata = json.loads(resp.read().decode("utf-8"))
                user = vdata.get("user") or {}
                minted_key = vdata.get("api_key") or ""
                save_config(
                    api_key=minted_key,
                    base_url=client.base_url,
                    auth_method="account",
                    email=user.get("email", email),
                    name=user.get("name", ""),
                    role=user.get("role", "developer"),
                    session_token=vdata.get("session_token", ""),
                )
                print(f"\n✓ Authenticated as \033[1m{user.get('email', email)}\033[0m ({user.get('role', 'developer')})")
                print(f"✓ CLI API key minted and saved locally to ~/.config/ornith/config.json ({vdata.get('key_prefix', 'sk-ornith-...')})")
                return 0
        except urllib.error.HTTPError as he:
            err_txt = he.read().decode("utf-8", errors="ignore")
            try:
                msg = json.loads(err_txt).get("error", {}).get("message", err_txt)
            except Exception:
                msg = err_txt
            print(f"✗ Verification failed: {msg}", file=sys.stderr)
            return 1

    elif choice == "2":
        raw_key = (given_api_key or "").strip()
        if not raw_key:
            try:
                raw_key = getpass.getpass("Enter your Ornith API Key (sk-ornith-...): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 1
        if not raw_key:
            print("Error: API key cannot be empty.", file=sys.stderr)
            return 1

        test_client = OrnithClient(base_url=client.base_url, api_key=raw_key)
        try:
            info = test_client.me()
            if not info.get("authenticated"):
                print("✗ Invalid or revoked API key.", file=sys.stderr)
                return 1
            user = info.get("user") or {}
            save_config(
                api_key=raw_key,
                base_url=client.base_url,
                auth_method="api_key",
                email=user.get("email", ""),
                name=user.get("name", ""),
                role=user.get("role", "developer"),
            )
            prefix = raw_key[:12] + "…" + raw_key[-4:] if len(raw_key) > 16 else "sk-ornith-…"
            print(f"\n✓ Authenticated via API Key ({prefix}) bound to \033[1m{user.get('email', 'account')}\033[0m ({user.get('role', 'developer')})")
            print("✓ Credentials saved locally to ~/.config/ornith/config.json")
            return 0
        except Exception as exc:
            print(f"✗ Failed to verify API key: {exc}", file=sys.stderr)
            return 1
    else:
        print("Invalid selection. Choose 1 (Account) or 2 (API Key).", file=sys.stderr)
        return 1


def handle_whoami(client: OrnithClient) -> int:
    cfg = load_config()
    try:
        info = client.me()
    except Exception as exc:
        print(f"✗ Could not reach Ornith Studio: {exc}", file=sys.stderr)
        return 1

    if not info.get("authenticated"):
        print("Not authenticated. Run `ornith login` to sign in via Developer Account or API Key.")
        return 1

    user = info.get("user") or {}
    key = client.api_key or ""
    prefix = key[:12] + "…" + key[-4:] if len(key) > 16 else "active"
    print(json.dumps({
        "authenticated": True,
        "email": user.get("email"),
        "name": user.get("name"),
        "role": user.get("role"),
        "auth_method": cfg.get("auth_method", "api_key"),
        "key_prefix": prefix,
        "rate_limit_rpm": user.get("rate_limit_rpm"),
        "model": DEFAULT_MODEL,
        "gateway": _display_endpoint(client.base_url),
        "storage": "local",
    }, indent=2))
    return 0


def handle_logout() -> int:
    for f in (CONFIG_FILE, KEY_FILE):
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass
    print("✓ Signed out and cleared local CLI credentials.")
    return 0


# ── Local Workspace Coding Agent (`ornith code`) ─────────────────────────

CODING_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the current workspace directory matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (default: '**/*')"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute path to the file"},
                    "start_line": {"type": "integer", "description": "1-based start line (optional)"},
                    "end_line": {"type": "integer", "description": "1-based end line (optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a local file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to write"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact target snippet in a local file with replacement text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "target": {"type": "string", "description": "Exact existing text to replace"},
                    "replacement": {"type": "string", "description": "New text"},
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a shell command in the local project directory (e.g., pytest, git diff, npm test).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
]


def _exec_local_coding_tool(name: str, args: Dict[str, Any], cwd: pathlib.Path) -> str:
    try:
        if name == "list_files":
            pat = str(args.get("pattern") or "**/*")
            matches = []
            for p in sorted(cwd.glob(pat)):
                if any(part.startswith(".") or part in ("node_modules", "__pycache__", "venv", ".venv") for part in p.parts):
                    continue
                if p.is_file():
                    try:
                        matches.append(str(p.relative_to(cwd)))
                    except Exception:
                        matches.append(str(p))
                if len(matches) >= 200:
                    break
            return "\n".join(matches) if matches else "(no matching files)"

        if name == "read_file":
            target = (cwd / str(args.get("path", ""))).resolve()
            if not target.exists():
                return f"Error: File not found: {args.get('path')}"
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            s = max(1, int(args.get("start_line") or 1))
            e = min(len(lines), int(args.get("end_line") or min(len(lines), s + 250)))
            numbered = [f"{i}: {lines[i - 1]}" for i in range(s, e + 1)]
            return "\n".join(numbered)

        if name == "write_file":
            rel = str(args.get("path", ""))
            target = (cwd / rel).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            content = str(args.get("content", ""))
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {rel}"

        if name == "edit_file":
            rel = str(args.get("path", ""))
            target = (cwd / rel).resolve()
            if not target.exists():
                return f"Error: File not found: {rel}"
            txt = target.read_text(encoding="utf-8")
            old = str(args.get("target", ""))
            new = str(args.get("replacement", ""))
            if old not in txt:
                return f"Error: Target string not found in {rel}"
            target.write_text(txt.replace(old, new, 1), encoding="utf-8")
            return f"Updated {rel} successfully."

        if name == "run_bash":
            cmd = str(args.get("command", ""))
            proc = subprocess.run(
                ["bash", "-lc", cmd],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            return f"Exit code: {proc.returncode}\n{out.strip()[:8000]}"

        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


def run_coding_agent(client: OrnithClient, initial_task: str = "", enable_thinking: bool = True) -> int:
    cwd = pathlib.Path.cwd()
    sys_prompt = (
        f"You are Ornith-1.5-35B Coding Agent operating inside `{cwd.name}`.\n"
        "Use `list_files`, `read_file`, `write_file`, `edit_file`, and `run_bash` to inspect code, implement features, "
        "fix bugs, and verify changes with tests. Be concise, technical, and accurate."
    )
    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_prompt}]

    def process_turn(user_input: str) -> None:
        messages.append({"role": "user", "content": user_input})
        for _ in range(10):
            msg = client.chat(
                messages,
                enable_thinking=enable_thinking,
                server_tools=False,
                tools=CODING_AGENT_TOOLS,
                stream=False,
            )
            if not isinstance(msg, dict):
                print(msg)
                messages.append({"role": "assistant", "content": str(msg)})
                return

            reasoning = msg.get("reasoning_content")
            if reasoning:
                print(f"\033[90m[thinking] {reasoning.strip()}\033[0m")

            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if content.strip():
                print(content.strip())

            messages.append({
                "role": "assistant",
                "content": content,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
            if not tool_calls:
                return

            for tc in tool_calls:
                fn = tc.get("function") or {}
                t_name = fn.get("name", "")
                try:
                    t_args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    t_args = {}
                print(f"\033[36m▸ {t_name}({json.dumps(t_args)[:100]})\033[0m")
                res = _exec_local_coding_tool(t_name, t_args, cwd)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or t_name,
                    "content": res,
                })

    if initial_task:
        process_turn(initial_task)
        return 0

    print("\033[1mOrnith-1.5-35B Interactive Coding Agent\033[0m")
    print(f"Workspace: {cwd} · Type `/help` for commands or `/exit` to quit.\n")
    while True:
        try:
            user_in = input("\033[32mornith>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_in:
            continue
        if user_in in ("/exit", "/quit", "exit", "quit"):
            break
        if user_in == "/help":
            print("Commands: /files (list workspace files), /whoami (account status), /clear (reset context), /exit")
            continue
        if user_in == "/clear":
            messages = [{"role": "system", "content": sys_prompt}]
            print("Context cleared.")
            continue
        if user_in == "/files":
            print(_exec_local_coding_tool("list_files", {}, cwd))
            continue
        if user_in == "/whoami":
            handle_whoami(client)
            continue
        process_turn(user_in)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ornith",
        description="Ornith-1.5-35B-Uncensored CLI & Coding Agent (Account/API Auth, Code Agent, Multimodal & Sandbox)",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt or subcommand (login | whoami | logout | code | sandbox | cmd | status | env)",
    )
    parser.add_argument("--url", default=None, help="Ornith Studio Gateway URL")
    parser.add_argument("--public", action="store_true", help="Use public HTTPS endpoint (https://kusi-qwen-api.vercel.app)")
    parser.add_argument("--account", "--email", dest="account", default=None, help="Authenticate via Developer Account email (Resend 6-digit code)")
    parser.add_argument("--code", dest="otp_code", default=None, help="6-digit Resend verification code for `ornith login`")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Authenticate or run with explicit API key (sk-ornith-...)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model preset name")
    parser.add_argument("--system", "-s", default=None, help="System prompt")
    parser.add_argument("--image", "-i", action="append", default=[], help="Attach image file(s)")
    parser.add_argument("--video", "-v", default=None, help="Attach video file (.mp4/.webm/.mov)")
    parser.add_argument("--no-think", action="store_true", help="Disable step-by-step thinking for faster replies")
    parser.add_argument("--no-tools", action="store_true", help="Disable built-in server tools")
    parser.add_argument("--lang", default="python", help="Language for `ornith sandbox` (python | node | bash)")
    parser.add_argument("--size", default="1024x1024", help="Dimensions for `ornith image` (e.g. 1024x1024)")
    parser.add_argument("--out", "-o", default=None, help="Output file path to save generated image (.png)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_intermixed_args()

    base_url = DEFAULT_PUBLIC_URL if args.public else args.url
    client = OrnithClient(base_url=base_url, api_key=args.api_key, model=args.model)

    sub = args.prompt[0] if args.prompt else ""

    if sub in ("login", "auth"):
        email_arg = args.account or (args.prompt[1] if len(args.prompt) > 1 and "@" in args.prompt[1] else None)
        return handle_login(client, email_arg, args.api_key, args.otp_code)

    if sub == "whoami" and len(args.prompt) == 1:
        return handle_whoami(client)

    if sub == "logout" and len(args.prompt) == 1:
        return handle_logout()

    if sub == "code":
        task_str = " ".join(args.prompt[1:]).strip()
        return run_coding_agent(client, initial_task=task_str, enable_thinking=not args.no_think)

    if sub in ("image", "img"):
        img_prompt = " ".join(args.prompt[1:]).strip()
        if not img_prompt:
            print("Usage: ornith image \"Detailed visual prompt\" [--out output.png]", file=sys.stderr)
            return 1
        res = client.generate_image(img_prompt, size=args.size, include_b64=bool(args.out))
        if args.out and res.get("data"):
            b64_str = res["data"][0].get("b64_json", "")
            if b64_str:
                out_p = pathlib.Path(args.out).expanduser()
                out_p.write_bytes(base64.b64decode(b64_str))
                res["data"][0].pop("b64_json", None)
                res["saved_to"] = str(out_p)
        print(json.dumps(res, indent=2))
        return 0

    if sub == "status" and len(args.prompt) == 1:
        st = client.status()
        print(json.dumps(st, indent=2))
        return 0

    if sub in ("env", "key") and len(args.prompt) == 1:
        if ENV_FILE.exists():
            print(ENV_FILE.read_text(encoding="utf-8").strip())
        else:
            print(f'export ORNITH_API_KEY="{client.api_key}"')
        return 0

    if sub == "sandbox":
        code = " ".join(args.prompt[1:]) or sys.stdin.read()
        res = client.run_sandbox(code, language=args.lang)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            if res.get("stdout"):
                sys.stdout.write(res["stdout"])
            if res.get("stderr"):
                sys.stderr.write(res["stderr"])
        return int(res.get("exit_code") or 0)

    if sub == "cmd":
        cmd_str = " ".join(args.prompt[1:]) or sys.stdin.read()
        res = client.run_command(cmd_str)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            if res.get("stdout"):
                sys.stdout.write(res["stdout"])
            if res.get("stderr"):
                sys.stderr.write(res["stderr"])
        return int(res.get("exit_code") or 0)

    prompt_text = " ".join(args.prompt).strip()
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            prompt_text = f"{prompt_text}\n\n```\n{stdin_text}\n```".strip()

    if not prompt_text and not args.image and not args.video:
        parser.print_help()
        return 1

    in_thinking = False
    for chunk in client.chat(
        prompt_text,
        system=args.system,
        images=args.image,
        video=args.video,
        enable_thinking=not args.no_think,
        server_tools=not args.no_tools,
        stream=True,
    ):
        if chunk["reasoning"]:
            if not in_thinking:
                sys.stderr.write("\033[90m[thinking] ")
                in_thinking = True
            sys.stderr.write(chunk["reasoning"])
            sys.stderr.flush()
        if chunk["content"]:
            if in_thinking:
                sys.stderr.write("\033[0m\n\n")
                sys.stderr.flush()
                in_thinking = False
            sys.stdout.write(chunk["content"])
            sys.stdout.flush()
    if in_thinking:
        sys.stderr.write("\033[0m\n")
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
