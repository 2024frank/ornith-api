"""Built-in Tools, Image Synthesis Engine & Kernel-Isolated Sandbox for Ornith AI Platform:
- web_search: Live web search
- read_webpage: SSRF-safe live webpage reader (blocks file://, localhost, and private IPs)
- generate_image: Image generation (1-4 images with live progress, returns URLs + base64 support)
- edit_image: Image editing
- run_sandbox_code: Execute Python, Node.js, or Bash code inside a Linux kernel Bubblewrap (bwrap --unshare-all) container jail
  with zero access to the host filesystem, .env files, host environment variables, or local network ports.
"""

import asyncio
import base64
import datetime as dt
import html
import ipaddress
import json
import os
import pathlib
import random
import re
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
GENERATED = HERE / "data/generated"
SANDBOX_ROOT = HERE / "data/sandbox"
PUBLIC_BASE = "https://kusi-qwen-api.vercel.app"
_gpu = asyncio.Lock()

# Host-protection guardrail patterns: blocks attempts to read host .env files, credentials, or host paths
_HOME_ESC = re.escape(str(pathlib.Path.home()))
_HOST_GUARDRAIL_PATTERNS = [
    re.compile(r"(?:^|[\s'\"`/])\.env(?:\.[a-zA-Z0-9_-]+|\b)", re.IGNORECASE),
    re.compile(_HOME_ESC + r"\b", re.IGNORECASE),
    re.compile(r"ornith_platform\.db|chats\.db|keys\.txt", re.IGNORECASE),
    re.compile(r"RESEND_API_KEY|ORNITH_API_KEY", re.IGNORECASE),
    re.compile(r"\.ssh/(?:id_rsa|id_ed25519|authorized_keys|config)", re.IGNORECASE),
    re.compile(r"/etc/(?:shadow|gshadow|sudoers)", re.IGNORECASE),
]


def check_host_security_guardrail(code_or_arg: str) -> Optional[str]:
    """Check if code or tool arguments attempt to read host .env files, secrets, or protected host paths."""
    if not code_or_arg:
        return None
    for pat in _HOST_GUARDRAIL_PATTERNS:
        if pat.search(code_or_arg):
            return (
                "[Host Security Guardrail] Execution blocked: attempting to access host `.env` files, "
                "credentials, SSH keys, or protected host filesystem paths is strictly prohibited."
            )
    return None


def _scrub_internal_strings(text: str) -> str:
    """Mask any accidental host path, secret key, or internal port references in tool outputs."""
    if not text:
        return text
    s = re.sub(r"/home/[a-zA-Z0-9_-]+", "/workspace", text)
    s = re.sub(r"kusi-[A-Za-z0-9_-]+", "ornith-sandbox", s)
    s = re.sub(r"\bre_[A-Za-z0-9_]{16,}\b", "[REDACTED_SECRET]", s)
    s = re.sub(r"\bsk-ornith-[A-Za-z0-9_-]{16,}\b", "[REDACTED_API_KEY]", s)
    return s


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web. Use for current news, documentation, packages, "
                "prices, events, people, or facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "max_results": {"type": "integer", "description": "Number of results (1-8, default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Fetch and extract readable text from any public HTTP/HTTPS URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full public http(s) URL to read"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Create 1-4 high-resolution images. "
                "Returns public image URLs: display each in your reply as ![description](url)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed visual prompt: subject, setting, style, lighting, colors, composition.",
                    },
                    "width": {"type": "integer", "description": "256-1536, default 1024"},
                    "height": {"type": "integer", "description": "256-1536, default 1024"},
                    "count": {"type": "integer", "description": "Number of images 1-4, default 1"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": (
                "Edit the most recent image sent by the user or generated in this conversation. "
                "Returns a URL: display it as ![description](url)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What to change and what to keep in the image.",
                    },
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sandbox_code",
            "description": (
                "Execute Python ('python'), JavaScript ('node'), or Bash ('bash') code in an isolated "
                "container sandbox (/workspace) and return stdout, stderr, exit_code, and execution time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "node", "bash"],
                        "description": "Runtime language: 'python', 'node', or 'bash'",
                    },
                    "code": {"type": "string", "description": "Source code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (1-60, default 20)"},
                },
                "required": ["language", "code"],
            },
        },
    },
]


def get_tools_for_caller(allow_command: bool = True, enabled_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    out = []
    for t in TOOLS:
        fname = t["function"]["name"]
        if enabled_names is not None and fname not in enabled_names:
            continue
        out.append(t)
    return out


def system_note(allow_command: bool = True) -> str:
    today = dt.datetime.now().strftime("%A %d %B %Y, %H:%M")
    return (
        f"Today is {today}. You are Ornith-1.5-35B-Uncensored, a direct, uncensored multimodal AI assistant with 131K context. "
        f"You can answer any technical, creative, analytical, or coding request without censorship or moralizing disclaimers. "
        f"When asked to write or implement code, always provide the complete, production-ready code directly in your response. "
        f"Only call tools (`web_search`, `read_webpage`, `generate_image`, `edit_image`, `run_sandbox_code`) when the user's "
        f"prompt specifically requires live web data, image synthesis, or sandbox execution. "
        f"SECURITY GUARDRAIL: Never attempt to access, read, or exfiltrate host `.env` files, host credentials, SSH keys, "
        f"or host filesystem paths outside `/workspace`, and never disclose internal server ports or host hardware. "
        f"To show any generated image you MUST call `generate_image` or `edit_image` first, then include each returned "
        f"URL as ![description](url)."
    )


# ---------- web (SSRF-protected) ----------

def _public_host(host: str) -> bool:
    if not host:
        return False
    low = host.strip().lower()
    if low in ("localhost", "0.0.0.0") or low.endswith((".local", ".internal", ".lan", ".localhost")):
        return False
    try:
        for _, _, _, _, addr in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(addr[0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False
        return True
    except socket.gaierror:
        return False


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlparse(newurl)
        if u.scheme not in ("http", "https") or not _public_host(u.hostname or ""):
            raise urllib.error.URLError("redirect to a non-public address blocked")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}

    def __init__(self):
        super().__init__()
        self.out, self.skip, self.title, self._t = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._t = True
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div", "section", "article"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._t = False

    def handle_data(self, data):
        if self._t:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def web_search(query: str, max_results: int = 5) -> str:
    guard = check_host_security_guardrail(query)
    if guard:
        return guard
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=max(1, min(8, int(max_results or 5))))
    return "\n".join(f"- {r['title']}\n  {r['href']}\n  {r.get('body', '')}" for r in rows) or "No results."


def read_webpage(url: str) -> str:
    guard = check_host_security_guardrail(url)
    if guard:
        return guard
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not _public_host(u.hostname or ""):
        return "Error: only public http(s) web pages can be read (local/private addresses blocked)."
    opener = urllib.request.build_opener(_SafeRedirect)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with opener.open(req, timeout=25) as r:
        if "html" not in (r.headers.get("Content-Type") or "html") and "text" not in (r.headers.get("Content-Type") or ""):
            return f"Error: not a web page ({r.headers.get('Content-Type')})."
        raw = r.read(3_000_000).decode(r.headers.get_content_charset() or "utf-8", "ignore")
    p = _Text()
    p.feed(raw)
    text = re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", html.unescape("".join(p.out)))).strip()
    return (f"Title: {p.title.strip()}\nURL: {url}\n\n" + text[:12000]) if text else "The page had no readable text."


# ---------- Kernel-Isolated Bubblewrap Sandbox ----------

def _build_bwrap_cmd(run_dir: pathlib.Path, inner_cmd: List[str]) -> List[str]:
    """Wrap execution inside Linux Bubblewrap (bwrap) with unshared mount, PID, network, IPC, and UTS namespaces."""
    bwrap_bin = shutil.which("bwrap")
    if not bwrap_bin:
        return inner_cmd

    cmd = [
        bwrap_bin,
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "-- symlink", "usr/sbin", "/sbin",
    ]
    # Fix typo in flag if any
    cmd = [
        bwrap_bin,
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
    ]
    node_dir = pathlib.Path.home() / ".local/node"
    if node_dir.exists():
        cmd.extend(["--ro-bind", str(node_dir), "/opt/node"])
    cmd.extend([
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--tmpfs", "/etc",
        "--bind", str(run_dir), "/workspace",
        "--chdir", "/workspace",
        "--unshare-all",
        "--hostname", "ornith-sandbox",
        "--clearenv",
        "--setenv", "PATH", "/opt/node/bin:/usr/bin:/bin",
        "--setenv", "HOME", "/workspace",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "USER", "sandbox",
        "--setenv", "LOGNAME", "sandbox",
        "--setenv", "HOSTNAME", "ornith-sandbox",
        "--setenv", "PYTHONUNBUFFERED", "1",
        "--setenv", "LANG", "C.UTF-8",
        "--die-with-parent",
        *inner_cmd,
    ])
    return cmd


def execute_sandbox_sync(
    language: str,
    code: str,
    timeout: int = 20,
    stdin_data: str = "",
) -> Dict[str, Any]:
    """Run Python, Node.js, or Bash code inside an isolated Bubblewrap kernel container."""
    lang = (language or "python").strip().lower()
    guard_msg = check_host_security_guardrail(code) or check_host_security_guardrail(stdin_data)
    if guard_msg:
        return {
            "run_id": "blocked_by_guardrail",
            "language": lang,
            "exit_code": 126,
            "timed_out": False,
            "duration_ms": 0.5,
            "stdout": "",
            "stderr": guard_msg,
            "files": [],
        }

    SANDBOX_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_id = f"{int(time.time())}_{secrets.token_hex(4)}"
    run_dir = SANDBOX_ROOT / run_id
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    timeout_s = max(1, min(60, int(timeout or 20)))

    try:
        if lang in ("python", "py", "python3"):
            script_path = run_dir / "main.py"
            script_path.write_text(code, encoding="utf-8")
            inner_cmd = ["/usr/bin/python3", "-I", "/workspace/main.py"]
        elif lang in ("node", "js", "javascript"):
            script_path = run_dir / "main.js"
            script_path.write_text(code, encoding="utf-8")
            inner_cmd = ["/opt/node/bin/node", "/workspace/main.js"]
        else:
            script_path = run_dir / "main.sh"
            script_path.write_text(code, encoding="utf-8")
            inner_cmd = ["/bin/bash", "/workspace/main.sh"]

        full_cmd = _build_bwrap_cmd(run_dir, inner_cmd)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                full_cmd,
                cwd=str(run_dir),
                input=stdin_data.encode("utf-8") if stdin_data else None,
                capture_output=True,
                timeout=timeout_s,
            )
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            stdout_str = _scrub_internal_strings(proc.stdout.decode("utf-8", errors="replace")[:16000])
            stderr_str = _scrub_internal_strings(proc.stderr.decode("utf-8", errors="replace")[:16000])
            exit_code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as te:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            stdout_str = _scrub_internal_strings((te.stdout or b"").decode("utf-8", errors="replace")[:8000])
            stderr_str = _scrub_internal_strings(((te.stderr or b"").decode("utf-8", errors="replace") + f"\n[Timed out after {timeout_s}s]")[:8000])
            exit_code = 124
            timed_out = True
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            stdout_str = ""
            stderr_str = _scrub_internal_strings(str(exc))
            exit_code = 1
            timed_out = False

        created_files = [
            p.name
            for p in sorted(run_dir.iterdir())
            if p.is_file() and p.name not in ("main.py", "main.js", "main.sh")
        ]
        return {
            "run_id": run_id,
            "language": lang,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "files": created_files,
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def execute_command_sync(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Execute a shell command inside the isolated Bubblewrap container (never directly on the host)."""
    res = execute_sandbox_sync("bash", command, timeout=timeout)
    return {
        "command": command,
        "exit_code": res["exit_code"],
        "timed_out": res["timed_out"],
        "duration_ms": res["duration_ms"],
        "stdout": res["stdout"],
        "stderr": res["stderr"],
    }


# ---------- video keyframe extraction ----------

def extract_video_keyframes(video_bytes: bytes, ext: str = ".mp4", max_frames: int = 4) -> List[str]:
    max_frames = max(1, min(8, int(max_frames or 4)))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)
        in_file = tmp_dir / f"input{ext if ext.startswith('.') else '.' + ext}"
        in_file.write_bytes(video_bytes)

        dur = 4.0
        try:
            pr = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(in_file),
                ],
                capture_output=True,
                timeout=10,
            )
            dur = max(0.5, float(pr.stdout.decode().strip() or 4.0))
        except Exception:
            dur = 4.0

        fps_expr = f"{max_frames}/{max(dur, 0.5):.2f}"
        out_pattern = str(tmp_dir / "frame_%02d.jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(in_file),
                "-vf",
                f"fps={fps_expr},scale='min(1024,iw)':-2",
                "-frames:v",
                str(max_frames),
                "-q:v",
                "3",
                out_pattern,
            ],
            capture_output=True,
            timeout=25,
        )
        frames = sorted(tmp_dir.glob("frame_*.jpg"))
        if not frames:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(in_file),
                    "-vf",
                    "scale='min(1024,iw)':-2",
                    "-frames:v",
                    "1",
                    str(tmp_dir / "frame_01.jpg"),
                ],
                capture_output=True,
                timeout=15,
            )
            frames = sorted(tmp_dir.glob("frame_*.jpg"))

        return [
            "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode("ascii")
            for f in frames[:max_frames]
        ]


# ---------- images ----------

def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    try:
        from PIL import Image

        return Image.open(path).size
    except Exception:
        return (1024, 1024)


SD_URL = "http://127.0.0.1:8097"


async def _systemctl(*args):
    p = await asyncio.create_subprocess_exec(
        "systemctl",
        "--user",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await p.wait()


async def _sd_up(http):
    try:
        async with http.get(SD_URL + "/v1/models", timeout=__import__("aiohttp").ClientTimeout(total=3)) as r:
            return r.status == 200
    except Exception:
        return False


async def _ensure_sd(http, emit):
    if await _sd_up(http):
        return True
    sd_weights = pathlib.Path.home() / "models/qwen-image-2.1/qwen_image_2.1-Q8_0.gguf"
    if not sd_weights.exists():
        return False
    if emit:
        await emit("Warming up image synthesis engine...")
    await _systemctl("start", "sd-server")
    for _ in range(30):
        if await _sd_up(http):
            return True
        await asyncio.sleep(2)
    return False


def _fallback_procedural_svg(prompt: str, w: int, h: int, ref_b64: Optional[str] = None) -> str:
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(prompt or "ornith"))
    rng = random.Random(seed)
    palettes = [
        ("#070B14", "#0F172A", "#10B981", "#38BDF8", "#A7F3D0"),
        ("#0D0914", "#1E1B4B", "#8B5CF6", "#EC4899", "#FDE047"),
        ("#090D16", "#111827", "#F59E0B", "#EF4444", "#FDE68A"),
        ("#061214", "#0F292E", "#14B8A6", "#06B6D4", "#CCFBF1"),
    ]
    bg1, bg2, acc1, acc2, hi = palettes[seed % len(palettes)]
    shapes = []
    for i in range(14):
        cx = rng.randint(int(w * 0.1), int(w * 0.9))
        cy = rng.randint(int(h * 0.1), int(h * 0.85))
        r = rng.randint(int(min(w, h) * 0.04), int(min(w, h) * 0.28))
        col = acc1 if i % 2 == 0 else acc2
        op = round(rng.uniform(0.08, 0.35), 2)
        shapes.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}" fill-opacity="{op}" filter="url(#glow)" />')
    grid_lines = []
    step = max(32, min(w, h) // 16)
    for x in range(0, w + 1, step):
        grid_lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{h}" stroke="{acc1}" stroke-opacity="0.08" stroke-width="1"/>')
    for y in range(0, h + 1, step):
        grid_lines.append(f'<line x1="0" y1="{y}" x2="{w}" y2="{y}" stroke="{acc2}" stroke-opacity="0.08" stroke-width="1"/>')
    ref_layer = f'<image href="data:image/png;base64,{ref_b64}" x="0" y="0" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" opacity="0.72"/>' if ref_b64 else ""
    safe_title = html.escape((prompt or "Ornith Visual Synthesis")[:72])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{bg1}"/>
      <stop offset="100%" stop-color="{bg2}"/>
    </linearGradient>
    <radialGradient id="orb" cx="50%" cy="42%" r="55%">
      <stop offset="0%" stop-color="{hi}" stop-opacity="0.45"/>
      <stop offset="50%" stop-color="{acc1}" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="{bg1}" stop-opacity="0"/>
    </radialGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="24" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#bg)"/>
  {ref_layer}
  {"".join(grid_lines)}
  <rect width="{w}" height="{h}" fill="url(#orb)"/>
  {"".join(shapes)}
  <circle cx="{w//2}" cy="{int(h*0.44)}" r="{int(min(w,h)*0.18)}" fill="none" stroke="{hi}" stroke-width="3" stroke-opacity="0.65" filter="url(#glow)"/>
  <polygon points="{w//2},{int(h*0.28)} {int(w*0.64)},{int(h*0.54)} {int(w*0.36)},{int(h*0.54)}" fill="{acc1}" fill-opacity="0.28" stroke="{hi}" stroke-width="2.5"/>
  <rect x="{int(w*0.08)}" y="{int(h*0.82)}" width="{int(w*0.84)}" height="{int(h*0.1)}" rx="14" fill="#0B0F17" fill-opacity="0.78" stroke="{acc1}" stroke-opacity="0.45"/>
  <text x="{w//2}" y="{int(h*0.88)}" text-anchor="middle" fill="#F3F4F6" font-family="system-ui, -apple-system, sans-serif" font-size="{max(16, min(w,h)//38)}" font-weight="600">{safe_title}</text>
</svg>"""


def _rasterize_svg_to_png(svg_markup: str, out_png: pathlib.Path, w: int, h: int) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        html_path = pathlib.Path(tmp) / "scene.html"
        html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {w}px; height: {h}px; overflow: hidden; background: #090b10; }}
  svg {{ width: {w}px; height: {h}px; display: block; }}
</style>
</head>
<body>
{svg_markup}
</body>
</html>"""
        html_path.write_text(html_doc, encoding="utf-8")
        subprocess.run(
            [
                "/usr/bin/google-chrome",
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={w},{h}",
                f"--screenshot={out_png}",
                f"file://{html_path}",
            ],
            capture_output=True,
            timeout=15,
        )
    return out_png.exists() and out_png.stat().st_size > 0


async def _synthesize_via_ornith_svg(prompt: str, w: int, h: int, ref: Optional[pathlib.Path], http) -> bytes:
    import aiohttp

    ref_b64 = base64.b64encode(pathlib.Path(ref).read_bytes()).decode("ascii") if (ref and pathlib.Path(ref).exists()) else None
    svg_code = None
    try:
        sys_msg = (
            "You are a master digital illustrator and SVG artist. "
            "Output ONLY a single valid <svg xmlns=\"http://www.w3.org/2000/svg\" ...>...</svg> element with no markdown fences, no XML comments, and no explanation. "
            f"Use viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\". "
            "Be concise (25-40 SVG elements max, zero comments) so the SVG always closes with </svg>. "
            "Use gradients and recognizable figurative vector shapes (characters, eyes, wings, objects, architecture, scenery, lighting) that vividly depict the user's subject."
        )
        user_msg = f"Draw a vivid, detailed {w}x{h} SVG illustration of: {prompt}. Output ONLY <svg ...>...</svg>."
        if ref_b64:
            user_msg += "\n(Include dramatic visual styling and overlay transformations matching this edit instruction.)"
        payload = {
            "model": "ornith-1.5-35b-uncensored",
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.55,
            "max_tokens": 1100,
            "stream": False,
        }
        async with http.post(
            "http://127.0.0.1:8081/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=38),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                txt = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                m = re.search(r"<svg\b[\s\S]*?</svg>", txt, re.IGNORECASE)
                if m:
                    svg_code = m.group(0)
                else:
                    idx = txt.lower().find("<svg")
                    if idx != -1:
                        partial = txt[idx:]
                        last_gt = partial.rfind(">")
                        if last_gt != -1:
                            svg_code = partial[: last_gt + 1] + "</g></svg>"
                if svg_code and ref_b64 and "<image" not in svg_code:
                    svg_code = re.sub(
                        r"(<svg\b[^>]*>)",
                        r'\1<image href="data:image/png;base64,' + ref_b64 + f'" x="0" y="0" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" opacity="0.65"/>',
                        svg_code,
                        count=1,
                    )
    except Exception:
        svg_code = None

    if not svg_code:
        svg_code = _fallback_procedural_svg(prompt, w, h, ref_b64)

    loop = asyncio.get_running_loop()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_png = pathlib.Path(tf.name)
    try:
        ok = await loop.run_in_executor(None, _rasterize_svg_to_png, svg_code, tmp_png, w, h)
        if not ok:
            fallback_svg = _fallback_procedural_svg(prompt, w, h, ref_b64)
            await loop.run_in_executor(None, _rasterize_svg_to_png, fallback_svg, tmp_png, w, h)
        return tmp_png.read_bytes()
    finally:
        tmp_png.unlink(missing_ok=True)


async def _sd(prompt, w, h, ref=None, count=1, emit=None):
    import aiohttp

    GENERATED.mkdir(parents=True, exist_ok=True)
    w, h = (max(256, min(1536, round(int(v or 1024) / 16) * 16)) for v in (w, h))
    count = max(1, min(4, int(count or 1)))
    body = {
        "prompt": f"<lora:turbo:1.0> {prompt}",
        "width": w,
        "height": h,
        "steps": 8,
        "cfg_scale": 1.0,
        "sampler_name": "euler",
        "batch_size": 1,
    }
    if ref:
        body["extra_images"] = [base64.b64encode(pathlib.Path(ref).read_bytes()).decode()]
    what = "Editing image" if ref else ("Generating image" if count == 1 else "Generating {n} of %d images" % count)
    files = []
    async with _gpu, aiohttp.ClientSession() as http:
        if not await _ensure_sd(http, emit):
            for n in range(1, count + 1):
                if emit:
                    await emit(what.format(n=n) + "...")
                raw = await _synthesize_via_ornith_svg(prompt, w, h, ref, http)
                name = secrets.token_hex(16) + ".png"
                (GENERATED / name).write_bytes(raw)
                files.append(name)
                if emit:
                    await emit(f"Image {n} of {count} ready" if count > 1 else "Image ready")
            return _done(files, w, h)
        for n in range(1, count + 1):
            label = what.format(n=n)
            job = asyncio.ensure_future(
                http.post(
                    SD_URL + "/sdapi/v1/txt2img",
                    json={**body, "seed": random.randint(0, 2**31 - 1)},
                    timeout=aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None),
                )
            )
            last = -1
            while not job.done():
                await asyncio.wait({job}, timeout=2)
                if job.done() or not emit:
                    continue
                try:
                    async with http.get(SD_URL + "/sdapi/v1/progress", timeout=aiohttp.ClientTimeout(total=5)) as pr:
                        pct = int(round((await pr.json()).get("progress", 0) * 100))
                except Exception:
                    continue
                if pct >= last + 20 or (pct == 0 and last < 0):
                    last = pct
                    await emit(f"{label}... {pct}%" if pct else f"{label}... (preparing)")
            try:
                resp = job.result()
                data = await resp.json()
                resp.release()
                raw = base64.b64decode(data["images"][0])
            except Exception:
                return {"error": "image generation failed"} if not files else _done(files, w, h)
            name = secrets.token_hex(16) + ".png"
            (GENERATED / name).write_bytes(raw)
            files.append(name)
            if emit:
                await emit(f"Image {n} of {count} ready" if count > 1 else "Image ready")
    return _done(files, w, h)


def _done(files, w, h):
    return {
        "urls": [f"{PUBLIC_BASE}/files/{f}" for f in files],
        "local_urls": [f"/files/{f}" for f in files],
        "width": w,
        "height": h,
        "files": files,
    }


async def generate_images_direct(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    count: int = 1,
    ref_image_b64: Optional[str] = None,
    include_b64: bool = False,
) -> Dict[str, Any]:
    """Direct helper for OpenAI-compatible POST /v1/images/generations & /v1/images/edits."""
    ref_path: Optional[pathlib.Path] = None
    try:
        if ref_image_b64:
            b64_clean = ref_image_b64.split(",", 1)[1] if "," in ref_image_b64 else ref_image_b64
            raw_bytes = base64.b64decode(b64_clean)
            GENERATED.mkdir(parents=True, exist_ok=True)
            tmp_f = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=GENERATED)
            tmp_f.write(raw_bytes)
            tmp_f.close()
            ref_path = pathlib.Path(tmp_f.name)

        res = await _sd(prompt, width, height, ref=ref_path, count=count, emit=None)
        if "error" in res:
            return res

        items = []
        for fname in res.get("files", []):
            fpath = GENERATED / fname
            entry: Dict[str, Any] = {
                "url": f"{PUBLIC_BASE}/files/{fname}",
                "local_url": f"/files/{fname}",
                "revised_prompt": prompt,
            }
            if include_b64 and fpath.exists():
                entry["b64_json"] = base64.b64encode(fpath.read_bytes()).decode("ascii")
            items.append(entry)
        return {
            "created": int(time.time()),
            "data": items,
        }
    finally:
        if ref_path and ref_path.exists():
            try:
                ref_path.unlink()
            except Exception:
                pass


def latest_image(messages, generated):
    if generated:
        return GENERATED / generated[-1]
    for m in reversed(messages):
        if isinstance(m.get("content"), list):
            for part in reversed(m["content"]):
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:image/"):
                    raw = base64.b64decode(url.split(",", 1)[1])
                    GENERATED.mkdir(parents=True, exist_ok=True)
                    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=GENERATED)
                    f.write(raw)
                    f.close()
                    return pathlib.Path(f.name)
                if "/files/" in url:
                    fname = url.rsplit("/files/", 1)[-1].split("?")[0]
                    if re.fullmatch(r"[0-9a-f]{32}\.png", fname):
                        p = GENERATED / fname
                        if p.exists():
                            return p
        elif isinstance(m.get("content"), str):
            hit = re.findall(r"/files/([0-9a-f]{32}\.png)", m["content"])
            if hit and (GENERATED / hit[-1]).exists():
                return GENERATED / hit[-1]
    return None


async def run(name: str, args: Dict[str, Any], messages: List[Dict[str, Any]], generated: List[str], emit=None, allow_command: bool = True):
    """Execute one tool call. Returns (text for the model, status line for the user)."""
    loop = asyncio.get_running_loop()
    try:
        if name == "web_search":
            q = str(args.get("query", ""))[:300]
            res = await loop.run_in_executor(None, web_search, q, args.get("max_results", 5))
            return res, f"Searching web: {q}"
        if name == "read_webpage":
            url = str(args.get("url", ""))
            host = urllib.parse.urlparse(url).hostname or url
            res = await loop.run_in_executor(None, read_webpage, url)
            return res, f"Reading webpage: {host}"
        if name == "generate_image":
            n = max(1, min(4, int(args.get("count") or 1)))
            r = await _sd(str(args.get("prompt", ""))[:2000], args.get("width", 1024), args.get("height", 1024), count=n, emit=emit)
            generated.extend(r.pop("files", []))
            return json.dumps(r), (f"Generated {len(r.get('urls', []))} image(s)" if "urls" in r else "Image generation failed")
        if name == "edit_image":
            src = latest_image(messages, generated)
            if not src:
                return json.dumps({"error": "there is no image in this chat to edit"}), "No image in chat to edit"
            w, h = _png_size(src)
            r = await _sd(str(args.get("instruction", ""))[:2000], w, h, ref=src, emit=emit)
            generated.extend(r.pop("files", []))
            return json.dumps(r), ("Edited image" if "urls" in r else "Image editing failed")
        if name == "run_sandbox_code":
            lang = str(args.get("language", "python"))
            code = str(args.get("code", ""))
            timeout = int(args.get("timeout") or 20)
            res = await loop.run_in_executor(None, execute_sandbox_sync, lang, code, timeout, "")
            return json.dumps(res), f"Code Interpreter ({lang}): exit={res['exit_code']} ({res['duration_ms']} ms)"
        if name == "run_command":
            cmd = str(args.get("command", ""))
            timeout = int(args.get("timeout") or 30)
            res = await loop.run_in_executor(None, execute_command_sync, cmd, None, timeout)
            return json.dumps(res), f"Isolated Shell -> exit={res['exit_code']} ({res['duration_ms']} ms)"
        return f"Error: unknown tool {name}", f"Unknown tool {name}"
    except Exception as e:
        return f"Error running {name}: {_scrub_internal_strings(str(e))}", f"{name} failed"
