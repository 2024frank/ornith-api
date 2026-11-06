# Ornith Studio & Multi-Protocol AI Platform (`ornith-1.5-35b-uncensored`)

An OpenAI Platform Studio + ChatGPT-grade developer console, multi-protocol inference gateway, image synthesis API, kernel-jailed code interpreter, and installable coding CLI (`ornith`) for **`ornith-1.5-35b-uncensored`** (35.5B Q8_0 + BF16 Multimodal Vision/Video Projector, 128K context window).

---

## Key Capabilities

- **OpenAI Studio + ChatGPT Web Console (`ui/index.html`)**
  - **Playground & Side-by-Side Compare**: Real-time SSE token streaming, collapsible chain-of-thought reasoning blocks, multimodal image/video attachments, system instructions, temperature/top-p sliders, response format (`text` / `json_object`), and saved Studio Presets.
  - **Code Interpreter Sandbox**: Interactive Python 3, Node.js, and Bash execution inside an isolated Linux kernel container.
  - **Traces & Logs Inspector**: Split-pane request trace table and detail inspector with per-request latency, token usage, and auth prefix.
  - **Storage & Image Generation Studio**: Browse generated assets and test image synthesis directly from the dashboard.
  - **API Keys & Team Access**: SHA-256 hashed secret keys (`sk-ornith-...`), per-key RPM rate limits, and passwordless 6-digit OTP email login via Resend (`2024frank@gmail.com` primary admin).
  - **Light & Dark Theme**: Adaptive OpenAI Studio dark & light palettes.

- **Multi-Protocol Wire Compatibility (`gateway.py`)**
  - `POST /v1/chat/completions` — OpenAI Chat Completions (streaming SSE, tool calling, multimodal vision/video, reasoning tokens)
  - `POST /v1/responses` — OpenAI Responses API (Codex CLI & Agents SDK compatible)
  - `POST /v1/messages` & `POST /v1/messages/count_tokens` — Anthropic Messages API (Claude Code CLI & Anthropic SDK compatible)
  - `POST /v1/completions` — Legacy OpenAI Text Completions
  - `POST /v1/images/generations` & `POST /v1/images/edits` — OpenAI-compatible Image Synthesis & Editing (`url` and `b64_json`)
  - `GET /v1/models` — Model discovery

- **Host-Protection Security Guardrails (`server_tools.py` & `gateway.py`)**
  - **Uncensored Model, Protected Host**: The model remains 100% uncensored for all user coding, security research, creative, and analytical tasks while strictly blocking any attempt to read or exfiltrate host `.env` files, API keys, SSH keys, or host filesystem paths.
  - **Linux Kernel Bubblewrap (`bwrap --unshare-all`) Container Jail**: Sandbox code runs with unshared network/PID/IPC/UTS namespaces, `--clearenv`, and empty `tmpfs` mounts over `/home`, `/etc`, and `/tmp`.
  - **SSRF & Secret Scrubbing**: Blocks `file://` URLs and loopback/private IP ranges in `read_webpage` and `image_url`, and scrubs internal host paths and secret prefixes from outputs.

- **Installable CLI & Autonomous Coding Agent (`ornith_sdk.py`)**
  - Install anywhere and export `PATH` in your active shell with one command:
    ```bash
    curl -fsSL https://kusi-qwen-api.vercel.app/install.sh | bash && export PATH="$HOME/.local/bin:$PATH"
    ```
  - Interactive dual-mode authentication (`ornith login`):
    1. **Developer Account (Email + 6-Digit Resend Code)** — automatically mints and saves a dedicated CLI API key.
    2. **Direct Secret API Key (`sk-ornith-...`)**

---

## Quickstart

### 1. Install & Authenticate the CLI

```bash
# Install the ornith CLI & export ~/.local/bin into your current terminal PATH
curl -fsSL https://kusi-qwen-api.vercel.app/install.sh | bash && export PATH="$HOME/.local/bin:$PATH"

# (If opening an older terminal tab where PATH wasn't reloaded yet, run:)
export PATH="$HOME/.local/bin:$PATH"

# Authenticate (choose [1] Email + 6-Digit Resend Code or [2] API Key)
ornith login

# Or sign in directly with your Developer Account email or API key:
ornith login --account 2024frank@gmail.com
ornith login --api-key sk-ornith-YOUR_SECRET_KEY

# Check account & connection status
ornith whoami
```

### 2. CLI Commands

```bash
# Stream a response
ornith "Write a lock-free ring buffer in C++20"

# Autonomous coding agent with local workspace file read/write/edit/bash tools
ornith code
ornith code "Refactor the auth module and run pytest"

# Multimodal vision & video analysis
ornith --image ui.png "Audit this interface for layout and contrast"
ornith --video demo.mp4 "Summarize the key events in this clip"

# Generate an image over the API and save locally
ornith image "A futuristic obsidian server vault lit by emerald neon" --out vault.png

# Execute code in the kernel-jailed remote sandbox
ornith sandbox -l python "import math; print([math.factorial(x) for x in range(8)])"

# Wire into Claude Code, OpenAI Codex, Aider, or Continue
eval "$(ornith env)"
```

### 3. OpenAI Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://kusi-qwen-api.vercel.app/v1",
    api_key="sk-ornith-YOUR_SECRET_KEY",
)

# Chat Completion (Streaming)
stream = client.chat.completions.create(
    model="ornith-1.5-35b-uncensored",
    messages=[
        {"role": "system", "content": "You are a principal systems engineer."},
        {"role": "user", "content": "Implement a token bucket rate limiter in Python."},
    ],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# Image Generation
img = client.images.generate(
    model="ornith-1.5-35b-uncensored",
    prompt="A cyberpunk owl perched on a terminal",
    size="1024x1024",
)
print("\nGenerated Image URL:", img.data[0].url)
```

---

## Self-Hosting Setup

1. Copy `.env.example` to `.env.secret` (`chmod 600 .env.secret`) and set your `RESEND_API_KEY`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `bwrap` (`bubblewrap`) is installed on Linux for kernel-jailed code execution.
4. Start the gateway:
   ```bash
   python3 gateway.py
   ```

