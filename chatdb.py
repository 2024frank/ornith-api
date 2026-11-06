"""Chat history for the Qwen API: every chat through the gateway is saved here, on this PC only.

  data/chats.db        SQLite: chats, feedback
  data/images/<sha>.*  images people sent (stored once, referenced from the chats as cj-image:<sha>)
The OpenAI API is stateless, so each request repeats the whole conversation. Requests are grouped into conversations
(X-Conversation-Id header if the client sends one, otherwise the key + first user message), and the training export
uses the longest version of each conversation."""
import base64, hashlib, json, pathlib, re, sqlite3, threading, time

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
IMAGES = DATA / "images"
DB = DATA / "chats.db"
_lock = threading.Lock()
EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif"}


def connect():
    DATA.mkdir(mode=0o700, exist_ok=True)
    IMAGES.mkdir(mode=0o700, exist_ok=True)
    db = sqlite3.connect(DB, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS chats (
      id TEXT PRIMARY KEY, ts REAL, key_name TEXT, conversation TEXT, path TEXT, stream INTEGER, status INTEGER,
      secs REAL, tokens_in INTEGER, tokens_out INTEGER, n_images INTEGER, train_ok INTEGER,
      request TEXT,      -- the request body, images replaced by cj-image:<sha>
      answer TEXT,       -- the assistant's reply
      reasoning TEXT,    -- its thinking, when thinking was on
      tool_calls TEXT);
    CREATE INDEX IF NOT EXISTS chats_key_ts ON chats(key_name, ts);
    CREATE INDEX IF NOT EXISTS chats_conv ON chats(conversation);
    CREATE TABLE IF NOT EXISTS feedback (
      chat_id TEXT, ts REAL, key_name TEXT, rating TEXT, correction TEXT, note TEXT);
    """)
    return db


def _store_images(messages):
    """Replace base64 data-URL images with cj-image:<sha> and save the bytes once. Returns the count."""
    n = 0
    for m in messages or []:
        if not isinstance(m.get("content"), list):
            continue
        for part in m["content"]:
            url = (part.get("image_url") or {}).get("url", "") if part.get("type") == "image_url" else ""
            mt = re.match(r"data:(image/[\w.+-]+);base64,", url)
            if not mt:
                continue
            raw = base64.b64decode(url[mt.end():], validate=False)
            sha = hashlib.sha256(raw).hexdigest()[:32]
            f = IMAGES / f"{sha}.{EXT.get(mt.group(1), 'bin')}"
            if not f.exists():
                f.write_bytes(raw)
                f.chmod(0o600)
            part["image_url"] = {**part["image_url"], "url": f"cj-image:{f.name}"}
            n += 1
    return n


def conversation_of(key_name, body, header):
    if header:
        return f"{key_name}:{header[:80]}"
    first = next((m for m in body.get("messages", []) if m.get("role") == "user"), None)
    seed = json.dumps(first.get("content") if first else body.get("prompt", ""), sort_keys=True)[:4000]
    return f"{key_name}:" + hashlib.sha1(seed.encode()).hexdigest()[:16]


def save(chat_id, key_name, train_ok, path, body, conversation, status, secs, stream, answer, reasoning, tool_calls, usage):
    body = json.loads(json.dumps(body))  # own copy: images get replaced
    n_img = _store_images(body.get("messages"))
    with _lock:
        db = connect()
        db.execute("INSERT OR REPLACE INTO chats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (chat_id, time.time(), key_name, conversation, path, int(stream), status, secs,
                    (usage or {}).get("prompt_tokens"), (usage or {}).get("completion_tokens"), n_img, int(train_ok),
                    json.dumps(body), answer, reasoning, json.dumps(tool_calls) if tool_calls else None))
        db.commit()
        db.close()


def from_response(data):
    """(answer, reasoning, tool_calls, usage) from a non-streamed OpenAI-style reply."""
    try:
        r = json.loads(data)
    except Exception:
        return None, None, None, {}
    c = (r.get("choices") or [{}])[0]
    msg = c.get("message") or {}
    return (msg.get("content") if "message" in c else c.get("text")), msg.get("reasoning_content"), msg.get("tool_calls"), r.get("usage", {})


class StreamCollector:
    """Rebuilds the reply from server-sent-event chunks while they are forwarded."""

    def __init__(self):
        self.buf, self.answer, self.reasoning, self.tools, self.usage = "", [], [], {}, {}

    def feed(self, chunk):
        self.buf += chunk.decode("utf-8", "replace")
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            try:
                ev = json.loads(line[5:])
            except Exception:
                continue
            self.usage = ev.get("usage") or self.usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    self.answer.append(d["content"])
                elif ch.get("text"):
                    self.answer.append(ch["text"])
                if d.get("reasoning_content"):
                    self.reasoning.append(d["reasoning_content"])
                for tc in d.get("tool_calls") or []:
                    t = self.tools.setdefault(tc.get("index", 0), {"id": tc.get("id"), "type": "function",
                                                                   "function": {"name": "", "arguments": ""}})
                    f = tc.get("function") or {}
                    t["function"]["name"] += f.get("name") or ""
                    t["function"]["arguments"] += f.get("arguments") or ""

    def result(self):
        return "".join(self.answer), "".join(self.reasoning) or None, list(self.tools.values()) or None, self.usage


def list_chats(key_name, limit=50, before=None):
    db = connect()
    rows = db.execute("SELECT id, ts, conversation, status, tokens_in, tokens_out, n_images, substr(answer,1,200) AS preview "
                      "FROM chats WHERE key_name=? AND ts < ? ORDER BY ts DESC LIMIT ?",
                      (key_name, before or 1e12, min(int(limit), 200))).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_chat(key_name, chat_id):
    db = connect()
    r = db.execute("SELECT * FROM chats WHERE key_name=? AND id=?", (key_name, chat_id)).fetchone()
    fb = db.execute("SELECT rating, correction, note, ts FROM feedback WHERE chat_id=? ORDER BY ts", (chat_id,)).fetchall()
    db.close()
    if not r:
        return None
    d = dict(r)
    d["request"] = json.loads(d["request"])
    d["tool_calls"] = json.loads(d["tool_calls"]) if d["tool_calls"] else None
    d["feedback"] = [dict(f) for f in fb]
    return d


def image_path(key_name, name):
    """Only images that appear in this key's own chats can be fetched."""
    if not re.fullmatch(r"[0-9a-f]{32}\.(png|jpg|webp|gif|bin)", name):
        return None
    db = connect()
    hit = db.execute("SELECT 1 FROM chats WHERE key_name=? AND request LIKE ? LIMIT 1", (key_name, f"%cj-image:{name}%")).fetchone()
    db.close()
    p = IMAGES / name
    return p if hit and p.exists() else None


def add_feedback(key_name, chat_id, rating, correction=None, note=None):
    db = connect()
    if not db.execute("SELECT 1 FROM chats WHERE key_name=? AND id=?", (key_name, chat_id)).fetchone():
        db.close()
        return False
    with _lock:
        db.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?)", (chat_id, time.time(), key_name, rating, correction, note))
        db.commit()
    db.close()
    return True
