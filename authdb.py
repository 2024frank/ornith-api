"""Security, Authentication (Resend 6-digit OTP), User Management & API Key Vault for Ornith-1.5-35B-Uncensored.

- Primary Administrator: 2024frank@gmail.com (seeded automatically, protected from deletion/demotion).
- Passwordless OTP Login via Resend API (PBKDF2-HMAC-SHA256 hashed codes, 10-min TTL, max 5 attempts, IP & email rate limits).
- Session Management: 256-bit cryptographically random session tokens stored as SHA-256 hashes in SQLite.
- API Keys: SHA-256 hashed keys (`sk-ornith-...`) with constant-time verification, per-key RPM rate limits,
  tool/command permissions, and seamless backwards compatibility with legacy `keys.txt`.
"""

import hashlib
import hmac
import json
import os
import pathlib
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
DB_PATH = DATA / "ornith_platform.db"
LEGACY_KEYS_PATH = HERE / "keys.txt"
SECRET_ENV_PATH = HERE / ".env.secret"
LOCAL_CONFIG_DIR = pathlib.Path.home() / ".config" / "ornith"
LOCAL_KEY_FILE = LOCAL_CONFIG_DIR / "api_key"
LOCAL_ENV_FILE = LOCAL_CONFIG_DIR / "env.sh"

PRIMARY_ADMIN_EMAIL = "2024frank@gmail.com"
OTP_TTL_SECONDS = 600          # 10 minutes
OTP_MAX_ATTEMPTS = 5
SESSION_TTL_SECONDS = 14 * 86400  # 14 days

_db_lock = threading.Lock()
_rate_buckets: Dict[str, List[float]] = {}
_legacy_mtime: float = 0.0


def _load_env_secret() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if SECRET_ENV_PATH.exists():
        for line in SECRET_ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pbkdf2_code(code: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        code.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def connect() -> sqlite3.Connection:
    DATA.mkdir(mode=0o700, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db() -> None:
    with _db_lock:
        db = connect()
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'developer',
                status TEXT NOT NULL DEFAULT 'active',
                rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
                allow_command_exec INTEGER NOT NULL DEFAULT 1,
                notes TEXT NOT NULL DEFAULT '',
                added_by TEXT NOT NULL DEFAULT 'system',
                created_at REAL NOT NULL,
                last_login_at REAL
            );

            CREATE TABLE IF NOT EXISTS otp_codes (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                expires_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                used INTEGER NOT NULL DEFAULT 0,
                request_ip TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_otp_email_created ON otp_codes(email, created_at);

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                token_hash TEXT UNIQUE NOT NULL,
                user_email TEXT NOT NULL,
                ip TEXT NOT NULL DEFAULT '',
                user_agent TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash);

            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                name TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
                allow_tools INTEGER NOT NULL DEFAULT 1,
                allow_command INTEGER NOT NULL DEFAULT 1,
                train_ok INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                last_used_at REAL,
                total_requests INTEGER NOT NULL DEFAULT 0,
                total_tokens_in INTEGER NOT NULL DEFAULT 0,
                total_tokens_out INTEGER NOT NULL DEFAULT 0,
                revoked INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
            CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_email);

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                actor TEXT NOT NULL,
                category TEXT NOT NULL,
                action TEXT NOT NULL,
                status INTEGER NOT NULL DEFAULT 200,
                latency_ms REAL NOT NULL DEFAULT 0,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0,
                ip TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(ts DESC);

            CREATE TABLE IF NOT EXISTS presets (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                system_prompt TEXT NOT NULL DEFAULT '',
                temperature REAL NOT NULL DEFAULT 0.6,
                top_p REAL NOT NULL DEFAULT 0.95,
                max_tokens INTEGER NOT NULL DEFAULT 4096,
                enable_thinking INTEGER NOT NULL DEFAULT 1,
                tools_json TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            );
            """
        )
        now = time.time()
        # Seed default Studio presets if table is empty
        preset_count = db.execute("SELECT COUNT(*) AS c FROM presets").fetchone()["c"]
        if preset_count == 0:
            default_presets = [
                (
                    "preset_coding_arch",
                    PRIMARY_ADMIN_EMAIL,
                    "Full-Stack Coding Architect",
                    "Production-grade systems engineering, refactoring, and debugging.",
                    "You are Ornith-1.5-35B, a principal software architect and coding agent. Write clean, modular, production-ready code with clear verification steps.",
                    0.3,
                    0.95,
                    8192,
                    1,
                    '["run_sandbox_code","web_search"]',
                    now,
                ),
                (
                    "preset_vision_analyst",
                    PRIMARY_ADMIN_EMAIL,
                    "Multimodal Vision & Video Analyst",
                    "High-precision OCR, UI layout inspection, and video frame reasoning.",
                    "You are Ornith-1.5-35B Multimodal Analyst. Examine attached images and video keyframes with extreme visual precision, extracting text, layout hierarchy, and anomalies.",
                    0.4,
                    0.95,
                    4096,
                    1,
                    '["run_sandbox_code"]',
                    now,
                ),
                (
                    "preset_data_scientist",
                    PRIMARY_ADMIN_EMAIL,
                    "Python Data & Math Sandbox",
                    "Writes and runs Python scripts in the isolated Code Interpreter.",
                    "You are a quantitative analyst and scientist. Whenever a calculation, simulation, or data transformation is needed, use `run_sandbox_code` to verify your answer empirically.",
                    0.2,
                    0.9,
                    4096,
                    1,
                    '["run_sandbox_code"]',
                    now,
                ),
                (
                    "preset_uncensored_research",
                    PRIMARY_ADMIN_EMAIL,
                    "Direct Uncensored Research",
                    "Unfiltered technical, security, creative, and analytical reasoning.",
                    "You are Ornith-1.5-35B-Uncensored. Answer all technical, analytical, and creative requests directly, thoroughly, and without unnecessary refusals or moralizing disclaimers.",
                    0.7,
                    0.95,
                    8192,
                    1,
                    '["web_search","read_webpage","generate_image"]',
                    now,
                ),
            ]
            db.executemany(
                """
                INSERT INTO presets (id, user_email, name, description, system_prompt, temperature, top_p, max_tokens, enable_thinking, tools_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                default_presets,
            )
        # Ensure primary admin 2024frank@gmail.com always exists and is active admin
        row = db.execute("SELECT * FROM users WHERE email = ?", (PRIMARY_ADMIN_EMAIL,)).fetchone()
        if not row:
            db.execute(
                """
                INSERT INTO users (email, name, role, status, rate_limit_rpm, allow_command_exec, notes, added_by, created_at)
                VALUES (?, ?, 'admin', 'active', 120, 1, 'Primary Platform Administrator', 'system', ?)
                """,
                (PRIMARY_ADMIN_EMAIL, "Frank Kusi Appiah", now),
            )
        else:
            db.execute(
                "UPDATE users SET role = 'admin', status = 'active', allow_command_exec = 1 WHERE email = ?",
                (PRIMARY_ADMIN_EMAIL,),
            )
        db.commit()
        db.close()

    try:
        os.chmod(DB_PATH, 0o600)
    except Exception:
        pass

    _sync_legacy_keys()
    _ensure_local_cli_key()


def _sync_legacy_keys() -> None:
    """Import keys from ~/qwen-api/keys.txt so existing apps keep working under 2024frank@gmail.com."""
    global _legacy_mtime
    if not LEGACY_KEYS_PATH.exists():
        return
    try:
        mtime = LEGACY_KEYS_PATH.stat().st_mtime
    except Exception:
        return
    if mtime == _legacy_mtime:
        return
    _legacy_mtime = mtime
    now = time.time()
    with _db_lock:
        db = connect()
        for line in LEGACY_KEYS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split("#")[0].split()
            if len(parts) < 2:
                continue
            k_name, raw_key = parts[0], parts[1]
            train_ok = 0 if "notrain" in parts[2:] else 1
            k_hash = _sha256_hex(raw_key)
            prefix = raw_key[:12] + "…" + raw_key[-4:] if len(raw_key) > 16 else raw_key[:8] + "…"
            existing = db.execute("SELECT id FROM api_keys WHERE key_hash = ?", (k_hash,)).fetchone()
            if not existing:
                db.execute(
                    """
                    INSERT INTO api_keys (id, user_email, name, key_prefix, key_hash, rate_limit_rpm, allow_tools, allow_command, train_ok, created_at)
                    VALUES (?, ?, ?, ?, ?, 60, 1, 1, ?, ?)
                    """,
                    (secrets.token_hex(8), PRIMARY_ADMIN_EMAIL, k_name, prefix, k_hash, train_ok, now),
                )
        db.commit()
        db.close()


def _ensure_local_cli_key() -> str:
    """Ensure ~/.config/ornith/api_key and env.sh exist with a valid active master key for local CLI & IDE agents."""
    LOCAL_CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if LOCAL_KEY_FILE.exists():
        raw = LOCAL_KEY_FILE.read_text(encoding="utf-8").strip()
        if raw:
            k_hash = _sha256_hex(raw)
            db = connect()
            row = db.execute("SELECT id, revoked FROM api_keys WHERE key_hash = ?", (k_hash,)).fetchone()
            db.close()
            if row and not row["revoked"]:
                _write_local_env_file(raw)
                return raw

    raw_key = "sk-ornith-" + secrets.token_urlsafe(32)
    k_hash = _sha256_hex(raw_key)
    prefix = raw_key[:14] + "…" + raw_key[-4:]
    now = time.time()
    with _db_lock:
        db = connect()
        db.execute(
            """
            INSERT INTO api_keys (id, user_email, name, key_prefix, key_hash, rate_limit_rpm, allow_tools, allow_command, train_ok, created_at)
            VALUES (?, ?, 'local-cli-master', ?, ?, 180, 1, 1, 1, ?)
            """,
            (secrets.token_hex(8), PRIMARY_ADMIN_EMAIL, prefix, k_hash, now),
        )
        db.commit()
        db.close()

    LOCAL_KEY_FILE.write_text(raw_key + "\n", encoding="utf-8")
    os.chmod(LOCAL_KEY_FILE, 0o600)
    _write_local_env_file(raw_key)
    return raw_key


def _write_local_env_file(raw_key: str) -> None:
    public_url = "https://kusi-qwen-api.vercel.app"
    env_content = (
        f'# Ornith-1.5-35B-Uncensored Local & Cloud API Environment\n'
        f'export ORNITH_API_KEY="{raw_key}"\n'
        f'export ORNITH_LOCAL_URL="http://127.0.0.1:8099"\n'
        f'export ORNITH_PUBLIC_URL="{public_url}"\n'
        f'export ORNITH_MODEL="ornith-1.5-35b-uncensored"\n'
        f'export OPENAI_API_KEY="{raw_key}"\n'
        f'export OPENAI_BASE_URL="http://127.0.0.1:8099/v1"\n'
        f'export ANTHROPIC_API_KEY="{raw_key}"\n'
        f'export ANTHROPIC_AUTH_TOKEN="{raw_key}"\n'
        f'export ANTHROPIC_BASE_URL="http://127.0.0.1:8099"\n'
    )
    LOCAL_ENV_FILE.write_text(env_content, encoding="utf-8")
    os.chmod(LOCAL_ENV_FILE, 0o600)


def check_rate_limit(bucket_key: str, max_per_window: int, window_secs: float = 60.0) -> Tuple[bool, int]:
    """Sliding-window rate limiter. Returns (allowed, remaining)."""
    now = time.time()
    hits = [t for t in _rate_buckets.get(bucket_key, []) if now - t < window_secs]
    if len(hits) >= max_per_window:
        _rate_buckets[bucket_key] = hits
        return False, 0
    hits.append(now)
    _rate_buckets[bucket_key] = hits
    return True, max(0, max_per_window - len(hits))


def record_audit(
    actor: str,
    category: str,
    action: str,
    status: int = 200,
    latency_ms: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    ip: str = "",
    detail: Any = "",
) -> None:
    detail_str = detail if isinstance(detail, str) else json.dumps(detail, default=str)
    with _db_lock:
        db = connect()
        db.execute(
            """
            INSERT INTO audit_logs (ts, actor, category, action, status, latency_ms, tokens_in, tokens_out, ip, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                actor[:120],
                category[:40],
                action[:120],
                int(status),
                round(float(latency_ms or 0.0), 1),
                int(tokens_in or 0),
                int(tokens_out or 0),
                (ip or "")[:64],
                detail_str[:2000],
            ),
        )
        db.commit()
        db.close()


# ── Resend Email Delivery ────────────────────────────────────────────────

def _send_resend_email(to_email: str, subject: str, html_body: str, text_body: str) -> Tuple[bool, str]:
    env = _load_env_secret()
    api_key = env.get("RESEND_API_KEY") or os.environ.get("RESEND_API_KEY", "")
    from_addr = env.get("RESEND_FROM") or os.environ.get("RESEND_FROM", "Ornith Developer Cloud <auth@communityhb.tech>")
    if not api_key:
        return False, "RESEND_API_KEY is not configured in .env.secret"

    payload = json.dumps(
        {
            "from": from_addr,
            "to": [to_email],
            "subject": subject,
            "html": html_body,
            "text": text_body,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OrnithDeveloperGateway/2.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            resp_data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return True, resp_data.get("id", "sent")
    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="ignore")
        return False, f"Resend HTTP {he.code}: {err_body[:240]}"
    except Exception as exc:
        return False, f"Resend error: {exc}"


def request_login_otp(email: str, request_ip: str = "") -> Tuple[bool, int, Dict[str, Any]]:
    """Generate a 6-digit OTP code for an allowlisted active user and send via Resend."""
    norm_email = (email or "").strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", norm_email):
        return False, 400, {"error": "Enter a valid email address."}

    allowed_ip, _ = check_rate_limit(f"otp_ip:{request_ip}", max_per_window=10, window_secs=900)
    if not allowed_ip:
        record_audit(norm_email, "auth", "otp_rate_limited_ip", 429, ip=request_ip)
        return False, 429, {"error": "Too many verification requests from this IP. Wait 15 minutes."}

    allowed_em, _ = check_rate_limit(f"otp_email:{norm_email}", max_per_window=5, window_secs=900)
    if not allowed_em:
        record_audit(norm_email, "auth", "otp_rate_limited_email", 429, ip=request_ip)
        return False, 429, {"error": "Too many verification codes requested for this email. Wait a few minutes."}

    db = connect()
    user = db.execute("SELECT * FROM users WHERE email = ?", (norm_email,)).fetchone()
    db.close()

    if not user:
        record_audit(norm_email, "auth", "otp_denied_unregistered", 403, ip=request_ip)
        return (
            False,
            403,
            {
                "error": (
                    "Access denied: this email is not registered on the Ornith Developer Platform. "
                    "Contact administrator 2024frank@gmail.com to be added."
                )
            },
        )

    if user["status"] != "active":
        record_audit(norm_email, "auth", "otp_denied_suspended", 403, ip=request_ip)
        return False, 403, {"error": "This developer account is currently suspended."}

    code = f"{secrets.randbelow(900000) + 100000:06d}"
    salt = secrets.token_hex(16)
    code_hash = _pbkdf2_code(code, salt)
    now = time.time()
    expires_at = now + OTP_TTL_SECONDS
    otp_id = secrets.token_hex(12)

    with _db_lock:
        db = connect()
        # Invalidate previous unused OTPs for this email
        db.execute("UPDATE otp_codes SET used = 1 WHERE email = ? AND used = 0", (norm_email,))
        db.execute(
            """
            INSERT INTO otp_codes (id, email, code_hash, salt, expires_at, attempts, used, request_ip, created_at)
            VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)
            """,
            (otp_id, norm_email, code_hash, salt, expires_at, request_ip or "", now),
        )
        db.commit()
        db.close()

    html_body = f"""
    <div style="background:#0D0E11;color:#ECECF1;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:36px 28px;border:1px solid #26282F;border-radius:10px;max-width:520px;margin:0 auto;">
      <div style="font-family:monospace;font-size:12px;color:#10A37F;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:12px;">
        ORNITH PLATFORM STUDIO
      </div>
      <h1 style="font-size:20px;font-weight:600;margin:0 0 12px;color:#FFFFFF;">
        Your 6-digit verification code
      </h1>
      <p style="font-size:14px;line-height:1.6;color:#A1A4B0;margin:0 0 24px;">
        Enter this one-time verification code to sign in to Ornith Studio &amp; CLI as <strong style="color:#ECECF1;">{norm_email}</strong> ({user['role']}):
      </p>
      <div style="background:#17181C;border:1px solid #2E313A;border-radius:8px;padding:20px;text-align:center;font-family:monospace;font-size:32px;font-weight:700;letter-spacing:0.28em;color:#10A37F;margin-bottom:24px;">
        {code}
      </div>
      <p style="font-size:12px;line-height:1.5;color:#6E7280;margin:0;">
        Expires in 10 minutes · Single-use PBKDF2-SHA256 verified.<br>
        If you did not request this code, you can safely ignore this email.
      </p>
    </div>
    """
    text_body = (
        f"Ornith-1.5-35B-Uncensored Developer Platform\n\n"
        f"Your 6-digit login code is: {code}\n\n"
        f"Account: {norm_email} ({user['role']})\n"
        f"Expires in 10 minutes."
    )

    sent_ok, msg_id = _send_resend_email(
        norm_email,
        f"{code} is your Ornith Developer Console login code",
        html_body,
        text_body,
    )
    if not sent_ok:
        record_audit(norm_email, "auth", "otp_send_failed", 502, ip=request_ip, detail=msg_id)
        return False, 502, {"error": f"Failed to send verification email via Resend: {msg_id}"}

    record_audit(norm_email, "auth", "otp_sent", 200, ip=request_ip, detail={"resend_id": msg_id})
    return (
        True,
        200,
        {
            "ok": True,
            "email": norm_email,
            "expires_in": OTP_TTL_SECONDS,
            "resend_id": msg_id,
            "message": f"Verification code sent to {norm_email} via Resend.",
        },
    )


def verify_login_otp(
    email: str,
    code: str,
    request_ip: str = "",
    user_agent: str = "",
) -> Tuple[bool, int, Dict[str, Any]]:
    """Verify 6-digit OTP code and create an authenticated session."""
    norm_email = (email or "").strip().lower()
    clean_code = re.sub(r"\s+", "", str(code or ""))
    if not norm_email or not re.fullmatch(r"\d{6}", clean_code):
        return False, 400, {"error": "Provide your email and the 6-digit numeric verification code."}

    allowed_ip, _ = check_rate_limit(f"verify_ip:{request_ip}", max_per_window=15, window_secs=600)
    if not allowed_ip:
        return False, 429, {"error": "Too many verification attempts. Please wait 10 minutes."}

    now = time.time()
    with _db_lock:
        db = connect()
        user = db.execute("SELECT * FROM users WHERE email = ?", (norm_email,)).fetchone()
        if not user or user["status"] != "active":
            db.close()
            return False, 403, {"error": "Account not authorized or suspended."}

        otp = db.execute(
            """
            SELECT * FROM otp_codes
            WHERE email = ? AND used = 0
            ORDER BY created_at DESC LIMIT 1
            """,
            (norm_email,),
        ).fetchone()

        if not otp:
            db.close()
            record_audit(norm_email, "auth", "otp_verify_missing", 401, ip=request_ip)
            return False, 401, {"error": "No active verification code found. Request a new code."}

        if now > otp["expires_at"]:
            db.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (otp["id"],))
            db.commit()
            db.close()
            record_audit(norm_email, "auth", "otp_verify_expired", 401, ip=request_ip)
            return False, 401, {"error": "Verification code has expired. Request a new code."}

        if otp["attempts"] >= OTP_MAX_ATTEMPTS:
            db.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (otp["id"],))
            db.commit()
            db.close()
            record_audit(norm_email, "auth", "otp_verify_locked", 401, ip=request_ip)
            return False, 401, {"error": "Maximum attempts exceeded. Request a new verification code."}

        candidate_hash = _pbkdf2_code(clean_code, otp["salt"])
        if not hmac.compare_digest(candidate_hash, otp["code_hash"]):
            new_attempts = otp["attempts"] + 1
            used_flag = 1 if new_attempts >= OTP_MAX_ATTEMPTS else 0
            db.execute(
                "UPDATE otp_codes SET attempts = ?, used = ? WHERE id = ?",
                (new_attempts, used_flag, otp["id"]),
            )
            db.commit()
            db.close()
            left = max(0, OTP_MAX_ATTEMPTS - new_attempts)
            record_audit(norm_email, "auth", "otp_verify_invalid", 401, ip=request_ip, detail={"attempts_left": left})
            return False, 401, {"error": f"Invalid verification code ({left} attempts remaining)."}

        # Valid code! Mark used, update user last_login_at, and issue session
        db.execute("UPDATE otp_codes SET used = 1 WHERE id = ?", (otp["id"],))
        db.execute("UPDATE users SET last_login_at = ? WHERE email = ?", (now, norm_email))

        raw_token = "ornith_sess_" + secrets.token_urlsafe(36)
        token_hash = _sha256_hex(raw_token)
        sess_id = secrets.token_hex(10)
        expires_at = now + SESSION_TTL_SECONDS
        db.execute(
            """
            INSERT INTO sessions (id, token_hash, user_email, ip, user_agent, created_at, expires_at, last_seen_at, revoked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (sess_id, token_hash, norm_email, (request_ip or "")[:64], (user_agent or "")[:200], now, expires_at, now),
        )
        db.commit()
        user_dict = dict(db.execute("SELECT * FROM users WHERE email = ?", (norm_email,)).fetchone())
        db.close()

    record_audit(norm_email, "auth", "login_success", 200, ip=request_ip)
    return (
        True,
        200,
        {
            "ok": True,
            "session_token": raw_token,
            "expires_at": expires_at,
            "user": {
                "email": user_dict["email"],
                "name": user_dict["name"],
                "role": user_dict["role"],
                "status": user_dict["status"],
                "rate_limit_rpm": user_dict["rate_limit_rpm"],
                "allow_command_exec": bool(user_dict["allow_command_exec"]),
            },
        },
    )


def get_session_user(raw_token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_token or not raw_token.startswith("ornith_sess_"):
        return None
    token_hash = _sha256_hex(raw_token.strip())
    now = time.time()
    db = connect()
    row = db.execute(
        """
        SELECT s.id AS session_id, s.expires_at, u.*
        FROM sessions s
        JOIN users u ON u.email = s.user_email
        WHERE s.token_hash = ? AND s.revoked = 0
        """,
        (token_hash,),
    ).fetchone()
    if not row:
        db.close()
        return None
    if now > row["expires_at"] or row["status"] != "active":
        db.close()
        return None
    db.close()
    return {
        "session_id": row["session_id"],
        "email": row["email"],
        "name": row["name"],
        "role": row["role"],
        "status": row["status"],
        "rate_limit_rpm": row["rate_limit_rpm"],
        "allow_command_exec": bool(row["allow_command_exec"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def revoke_session(raw_token: Optional[str]) -> None:
    if not raw_token:
        return
    token_hash = _sha256_hex(raw_token.strip())
    with _db_lock:
        db = connect()
        db.execute("UPDATE sessions SET revoked = 1 WHERE token_hash = ?", (token_hash,))
        db.commit()
        db.close()


# ── API Key Verification & Management ────────────────────────────────────

def verify_api_key(given_key: str) -> Optional[Dict[str, Any]]:
    """Verify an API key (`sk-ornith-...` or legacy key) or an active session token (`ornith_sess_...`)."""
    if not given_key:
        return None
    given_key = given_key.strip()

    # Allow logged-in Developer Console Playground to call /v1/* directly with its session token
    if given_key.startswith("ornith_sess_"):
        sess_user = get_session_user(given_key)
        if not sess_user:
            return None
        return {
            "key_id": "session",
            "key_name": f"console:{sess_user['email']}",
            "user_email": sess_user["email"],
            "role": sess_user["role"],
            "rate_limit_rpm": int(sess_user["rate_limit_rpm"]),
            "allow_tools": True,
            "allow_command": bool(sess_user["allow_command_exec"]),
            "train_ok": True,
        }

    _sync_legacy_keys()
    given_hash = _sha256_hex(given_key)
    db = connect()
    rows = db.execute(
        """
        SELECT k.*, u.status AS user_status, u.role AS user_role, u.allow_command_exec AS user_allow_cmd
        FROM api_keys k
        JOIN users u ON u.email = k.user_email
        WHERE k.revoked = 0
        """
    ).fetchall()
    db.close()

    for r in rows:
        if hmac.compare_digest(given_hash, r["key_hash"]):
            if r["user_status"] != "active":
                return None
            return {
                "key_id": r["id"],
                "key_name": r["name"],
                "user_email": r["user_email"],
                "role": r["user_role"],
                "rate_limit_rpm": int(r["rate_limit_rpm"] or 60),
                "allow_tools": bool(r["allow_tools"]),
                "allow_command": bool(r["allow_command"] and r["user_allow_cmd"]),
                "train_ok": bool(r["train_ok"]),
            }
    return None


def record_key_usage(key_id: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
    if not key_id or key_id == "session":
        return
    with _db_lock:
        db = connect()
        db.execute(
            """
            UPDATE api_keys
            SET last_used_at = ?,
                total_requests = total_requests + 1,
                total_tokens_in = total_tokens_in + ?,
                total_tokens_out = total_tokens_out + ?
            WHERE id = ?
            """,
            (time.time(), int(tokens_in or 0), int(tokens_out or 0), key_id),
        )
        db.commit()
        db.close()


def list_api_keys(user_email: str, is_admin: bool = False) -> List[Dict[str, Any]]:
    db = connect()
    if is_admin:
        rows = db.execute(
            "SELECT id, user_email, name, key_prefix, rate_limit_rpm, allow_tools, allow_command, train_ok, created_at, last_used_at, total_requests, total_tokens_in, total_tokens_out, revoked FROM api_keys ORDER BY revoked ASC, created_at DESC"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, user_email, name, key_prefix, rate_limit_rpm, allow_tools, allow_command, train_ok, created_at, last_used_at, total_requests, total_tokens_in, total_tokens_out, revoked FROM api_keys WHERE user_email = ? ORDER BY revoked ASC, created_at DESC",
            (user_email,),
        ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def create_api_key(
    user_email: str,
    name: str,
    rate_limit_rpm: int = 60,
    allow_tools: bool = True,
    allow_command: bool = True,
    train_ok: bool = True,
) -> Dict[str, Any]:
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", (name or "key").strip())[:48] or "default-key"
    raw_key = "sk-ornith-" + secrets.token_urlsafe(32)
    k_hash = _sha256_hex(raw_key)
    prefix = raw_key[:14] + "…" + raw_key[-4:]
    key_id = secrets.token_hex(8)
    now = time.time()
    rpm = max(5, min(600, int(rate_limit_rpm or 60)))

    with _db_lock:
        db = connect()
        db.execute(
            """
            INSERT INTO api_keys (id, user_email, name, key_prefix, key_hash, rate_limit_rpm, allow_tools, allow_command, train_ok, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key_id, user_email, clean_name, prefix, k_hash, rpm, int(allow_tools), int(allow_command), int(train_ok), now),
        )
        db.commit()
        db.close()

    return {
        "id": key_id,
        "name": clean_name,
        "user_email": user_email,
        "key": raw_key,
        "key_prefix": prefix,
        "rate_limit_rpm": rpm,
        "allow_tools": bool(allow_tools),
        "allow_command": bool(allow_command),
        "train_ok": bool(train_ok),
        "created_at": now,
    }


def revoke_api_key(key_id: str, actor_email: str, is_admin: bool = False) -> bool:
    with _db_lock:
        db = connect()
        if is_admin:
            cur = db.execute("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,))
        else:
            cur = db.execute("UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_email = ?", (key_id, actor_email))
        db.commit()
        changed = cur.rowcount > 0
        db.close()
    return changed


# ── Admin User Management ────────────────────────────────────────────────

def list_users() -> List[Dict[str, Any]]:
    db = connect()
    rows = db.execute(
        """
        SELECT u.*,
               (SELECT COUNT(*) FROM api_keys k WHERE k.user_email = u.email AND k.revoked = 0) AS active_keys,
               (SELECT COALESCE(SUM(k.total_requests), 0) FROM api_keys k WHERE k.user_email = u.email) AS total_requests,
               (SELECT COALESCE(SUM(k.total_tokens_in + k.total_tokens_out), 0) FROM api_keys k WHERE k.user_email = u.email) AS total_tokens
        FROM users u
        ORDER BY CASE WHEN u.email = ? THEN 0 WHEN u.role = 'admin' THEN 1 ELSE 2 END, u.created_at ASC
        """,
        (PRIMARY_ADMIN_EMAIL,),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def add_or_update_user(
    email: str,
    name: str = "",
    role: str = "developer",
    status: str = "active",
    rate_limit_rpm: int = 60,
    allow_command_exec: bool = True,
    notes: str = "",
    added_by: str = PRIMARY_ADMIN_EMAIL,
    send_invite: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    norm_email = (email or "").strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", norm_email):
        return False, "Invalid email address.", {}

    if norm_email == PRIMARY_ADMIN_EMAIL:
        role = "admin"
        status = "active"
        allow_command_exec = True

    if role not in ("admin", "developer"):
        role = "developer"
    if status not in ("active", "suspended"):
        status = "active"
    rpm = max(5, min(600, int(rate_limit_rpm or 60)))
    now = time.time()

    with _db_lock:
        db = connect()
        existing = db.execute("SELECT * FROM users WHERE email = ?", (norm_email,)).fetchone()
        if existing:
            db.execute(
                """
                UPDATE users
                SET name = COALESCE(NULLIF(?, ''), name),
                    role = ?,
                    status = ?,
                    rate_limit_rpm = ?,
                    allow_command_exec = ?,
                    notes = COALESCE(NULLIF(?, ''), notes)
                WHERE email = ?
                """,
                (name.strip(), role, status, rpm, int(allow_command_exec), notes.strip(), norm_email),
            )
        else:
            db.execute(
                """
                INSERT INTO users (email, name, role, status, rate_limit_rpm, allow_command_exec, notes, added_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (norm_email, name.strip() or norm_email.split("@")[0], role, status, rpm, int(allow_command_exec), notes.strip(), added_by, now),
            )
        db.commit()
        user_row = dict(db.execute("SELECT * FROM users WHERE email = ?", (norm_email,)).fetchone())
        db.close()

    invite_status = "not_requested"
    if send_invite and status == "active":
        html_body = f"""
        <div style="background:#0D0E11;color:#ECECF1;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:36px 28px;border:1px solid #26282F;border-radius:10px;max-width:520px;margin:0 auto;">
          <div style="font-family:monospace;font-size:12px;color:#10A37F;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:12px;">
            ORNITH PLATFORM STUDIO
          </div>
          <h1 style="font-size:20px;font-weight:600;margin:0 0 12px;color:#FFFFFF;">
            Your developer account is active
          </h1>
          <p style="font-size:14px;line-height:1.6;color:#A1A4B0;margin:0 0 20px;">
            <strong style="color:#ECECF1;">{added_by}</strong> granted <strong style="color:#ECECF1;">{norm_email}</strong> ({role}) access to the Ornith-1.5-35B-Uncensored Developer Platform &amp; Coding CLI.
          </p>
          <p style="font-size:14px;line-height:1.6;color:#A1A4B0;margin:0 0 24px;">
            Sign in with your email address at <a href="https://kusi-qwen-api.vercel.app" style="color:#10A37F;">https://kusi-qwen-api.vercel.app</a> or via <code>ornith login</code> to receive a 6-digit login code, mint API keys, and use the Playground &amp; Coding Agent.
          </p>
        </div>
        """
        ok, info = _send_resend_email(
            norm_email,
            "You have been invited to Ornith Platform Studio",
            html_body,
            f"Your developer account ({norm_email}) has been activated on https://kusi-qwen-api.vercel.app by {added_by}.",
        )
        invite_status = f"sent ({info})" if ok else f"failed ({info})"

    user_row["invite_status"] = invite_status
    return True, "User saved.", user_row


def delete_user(email: str) -> Tuple[bool, str]:
    norm_email = (email or "").strip().lower()
    if norm_email == PRIMARY_ADMIN_EMAIL:
        return False, "Cannot delete the primary platform administrator (2024frank@gmail.com)."
    with _db_lock:
        db = connect()
        db.execute("UPDATE api_keys SET revoked = 1 WHERE user_email = ?", (norm_email,))
        db.execute("UPDATE sessions SET revoked = 1 WHERE user_email = ?", (norm_email,))
        cur = db.execute("DELETE FROM users WHERE email = ?", (norm_email,))
        db.commit()
        deleted = cur.rowcount > 0
        db.close()
    return (True, "User deleted and keys revoked.") if deleted else (False, "User not found.")


def list_audit_logs(limit: int = 80, actor_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    db = connect()
    lim = max(1, min(300, int(limit or 80)))
    if actor_filter:
        rows = db.execute(
            "SELECT * FROM audit_logs WHERE actor LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{actor_filter}%", lim),
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM audit_logs ORDER BY ts DESC LIMIT ?", (lim,)).fetchall()
    db.close()
    return [dict(r) for r in rows]


def list_presets(user_email: str) -> List[Dict[str, Any]]:
    db = connect()
    rows = db.execute(
        "SELECT * FROM presets WHERE user_email = ? OR user_email = ? ORDER BY created_at ASC",
        (user_email, PRIMARY_ADMIN_EMAIL),
    ).fetchall()
    db.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["tools"] = json.loads(d.get("tools_json") or "[]")
        except Exception:
            d["tools"] = []
        d["enable_thinking"] = bool(d.get("enable_thinking", 1))
        out.append(d)
    return out


def save_preset(
    user_email: str,
    name: str,
    description: str = "",
    system_prompt: str = "",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    enable_thinking: bool = True,
    tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    pid = "preset_" + secrets.token_hex(6)
    now = time.time()
    clean_name = (name or "Custom Preset").strip()[:64]
    tools_json = json.dumps(tools if isinstance(tools, list) else [])
    with _db_lock:
        db = connect()
        db.execute(
            """
            INSERT INTO presets (id, user_email, name, description, system_prompt, temperature, top_p, max_tokens, enable_thinking, tools_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                user_email,
                clean_name,
                (description or "").strip()[:160],
                (system_prompt or "").strip()[:8000],
                round(float(temperature or 0.6), 2),
                round(float(top_p or 0.95), 2),
                int(max_tokens or 4096),
                1 if enable_thinking else 0,
                tools_json,
                now,
            ),
        )
        db.commit()
        db.close()
    return {
        "id": pid,
        "user_email": user_email,
        "name": clean_name,
        "description": (description or "").strip()[:160],
        "system_prompt": (system_prompt or "").strip()[:8000],
        "temperature": round(float(temperature or 0.6), 2),
        "top_p": round(float(top_p or 0.95), 2),
        "max_tokens": int(max_tokens or 4096),
        "enable_thinking": bool(enable_thinking),
        "tools": tools if isinstance(tools, list) else [],
        "created_at": now,
    }


def delete_preset(preset_id: str, user_email: str, is_admin: bool = False) -> bool:
    with _db_lock:
        db = connect()
        if is_admin:
            cur = db.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
        else:
            cur = db.execute("DELETE FROM presets WHERE id = ? AND user_email = ?", (preset_id, user_email))
        db.commit()
        ok = cur.rowcount > 0
        db.close()
    return ok


