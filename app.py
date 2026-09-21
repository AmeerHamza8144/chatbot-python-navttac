"""Headless LiveKit helpers for the HamzaLive voice agent.

The browser/Gradio presentation layer has been removed. Run ``agent.py``
to start the LiveKit worker; this module keeps the server-side worker and
token helpers available for another API or frontend.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from livekit import api


load_dotenv(Path(__file__).with_name(".env"))

PROJECT_DIR = Path(__file__).resolve().parent
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "my-agent").strip() or "my-agent"
DEFAULT_ROOM = f"voice-agent-{uuid.uuid4().hex[:12]}"
DEFAULT_IDENTITY = "Hamza"

_worker_process: subprocess.Popen[bytes] | None = None
_worker_lock = threading.Lock()


def _env_ready() -> bool:
    return all(
        os.getenv(key, "").strip()
        for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    )


def _worker_is_running() -> bool:
    return _worker_process is not None and _worker_process.poll() is None


def _configuration_summary() -> str:
    url = os.getenv("LIVEKIT_URL", "").strip()
    if _env_ready():
        return f"Ready · {url}"

    missing = [
        label
        for key, label in (
            ("LIVEKIT_URL", "URL"),
            ("LIVEKIT_API_KEY", "API key"),
            ("LIVEKIT_API_SECRET", "API secret"),
        )
        if not os.getenv(key, "").strip()
    ]
    return "Missing " + ", ".join(missing)


def worker_status_markdown() -> str:
    running = _worker_is_running()
    state = "ONLINE" if running else "OFFLINE"
    pid = str(_worker_process.pid) if running and _worker_process else "—"
    return f"<span>{state} · PID {pid} · {_configuration_summary()}</span>"


def start_worker() -> tuple[str, str]:
    """Start the LiveKit worker defined in ``agent.py``."""

    global _worker_process

    if not _env_ready():
        return (
            worker_status_markdown(),
            "Worker not started. Add LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET to .env.",
        )

    with _worker_lock:
        if _worker_is_running():
            return worker_status_markdown(), "Worker is already online."

        command = [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['agent-worker', 'start']; from agent import agents, server; agents.cli.run_app(server)",
        ]
        worker_env = os.environ.copy()
        worker_env["PYTHONUNBUFFERED"] = "1"
        # Do not inherit the unavailable local proxy used by this workspace.
        # It makes LiveKit Cloud resolve to 127.0.0.1:9 instead of the cloud URL.
        for proxy_key in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ):
            if worker_env.get(proxy_key, "").strip().lower() in {
                "http://127.0.0.1:9",
                "http://localhost:9",
            }:
                worker_env.pop(proxy_key, None)
        worker_env["NO_PROXY"] = "*"
        worker_env["no_proxy"] = "*"

        try:
            _worker_process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=worker_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as exc:
            _worker_process = None
            return worker_status_markdown(), f"Could not start worker: {exc}"

    return worker_status_markdown(), f"Worker start requested (PID {_worker_process.pid})."


def stop_worker() -> tuple[str, str]:
    """Stop only the worker process started by this module."""

    global _worker_process

    with _worker_lock:
        process = _worker_process
        if process is None or process.poll() is not None:
            _worker_process = None
            return worker_status_markdown(), "Worker is already offline."

        try:
            process.terminate()
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        finally:
            _worker_process = None

    return worker_status_markdown(), "Worker stopped."


def _create_token(room_name: str, participant_identity: str) -> str:
    """Create a browser token that dispatches the configured agent."""

    access_token = (
        api.AccessToken(
            os.getenv("LIVEKIT_API_KEY", ""),
            os.getenv("LIVEKIT_API_SECRET", ""),
        )
        .with_identity(participant_identity)
        .with_name(participant_identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)],
            )
        )
    )
    return access_token.to_jwt()


def start_session(room_name: str, participant_identity: str) -> tuple[str, str, str, str]:
    """Start the worker and issue a short-lived browser room token."""

    room_name = (room_name or "").strip()
    participant_identity = (participant_identity or "").strip()
    livekit_url = os.getenv("LIVEKIT_URL", "").strip().rstrip("/")

    if not _env_ready():
        return "", "", "Not connected", "Add the LiveKit credentials to .env before connecting."
    if not room_name or not participant_identity:
        return "", "", "Not connected", "Room name and participant identity are required."
    if not livekit_url.startswith(("ws://", "wss://", "http://", "https://")):
        return "", "", "Not connected", "LIVEKIT_URL must start with ws://, wss://, http://, or https://."

    _worker_status, worker_message = start_worker()
    if not _worker_is_running():
        return "", "", "Not connected", f"The backend worker could not start. {worker_message}"

    try:
        token = _create_token(room_name, participant_identity)
    except Exception as exc:
        stop_worker()
        return "", "", "Connection error", f"Could not create a LiveKit token: {exc}"

    return token, livekit_url, f"Ready · {room_name}", f"Token issued for {participant_identity}. Backend worker is online."


def disconnect_session() -> tuple[str, str, str, str]:
    """Stop the worker and clear the current browser session values."""

    stop_worker()
    return "", "", "Disconnected", "The browser session ended and the backend worker is offline."


def _shutdown_worker() -> None:
    if _worker_is_running():
        stop_worker()


atexit.register(_shutdown_worker)


APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --hl-bg: #ffffff;
    --hl-surface: #ffffff;
    --hl-surface-2: #f8fafc;
    --hl-border: #e5e7eb;
    --hl-muted: #6b7280;
    --hl-text: #111827;
    --hl-blue: #2563eb;
    --hl-purple: #9333ea;
    --hl-cyan: #06b6d4;
    --hl-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}

* { box-sizing: border-box; }
html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    overflow-x: hidden !important;
}
body {
    height: 100vh !important;
    overflow-y: hidden !important;
    background: var(--hl-bg) !important;
    color: var(--hl-text) !important;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}
/* Strip every layer Gradio wraps our HTML in so nothing centers or pads it */
gradio-app,
.gradio-container,
.gradio-container .main,
.gradio-container > div,
.contain,
#root,
.app {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    max-width: none !important;
    height: 100% !important;
    min-height: 100vh !important;
    background: var(--hl-bg) !important;
}
.hl-shell {
    width: 100%;
    height: 100vh;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 16px 24px;
}
.hl-app-header {
    width: 100%;
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    background: #ffffff;
    border-bottom: 1px solid var(--hl-border);
    padding-bottom: 12px;
}
.hl-logo {
    width: 32px;
    height: 32px;
    display: grid;
    place-items: center;
    padding: 2px;
    border-radius: 8px;
    background: linear-gradient(135deg, #2563eb, #9333ea, #06b6d4);
}
.hl-logo-inner {
    width: 100%;
    height: 100%;
    display: grid;
    place-items: center;
    border-radius: 6px;
    background: #ffffff;
    color: var(--hl-blue);
    font-size: 16px;
    font-weight: 800;
}
.hl-status-dot.live {
    background: var(--hl-blue);
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14);
    animation: hl-pulse 2s infinite;
}
@keyframes hl-pulse {
    0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(37, 99, 235, 0); }
    100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}
.hl-panel {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--hl-border);
    border-radius: 16px;
    background: #ffffff;
    box-shadow: var(--hl-shadow);
    min-height: 0;
}
.hl-workspace-split {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    min-height: 0;
    overflow: hidden;
}
.hl-visualizer {
    position: relative;
    flex: 1;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    overflow: hidden;
    padding: 16px;
    background: radial-gradient(circle at 50% 28%, #eff6ff, transparent 42%), linear-gradient(180deg, #ffffff, #f8fafc);
    border-right: 1px solid var(--hl-border);
}
.hl-ambient-glow {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: radial-gradient(circle at center, rgba(37, 99, 235, 0.12), transparent 68%);
}
#gemini-canvas {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    cursor: pointer;
    transition: transform .3s ease;
}
#gemini-canvas:hover { transform: scale(1.04); }
.hl-transcript-side {
    flex: 1;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 16px;
    background: #f9fafb;
    min-height: 0;
}
.hl-transcript-feed {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    border: 1px solid var(--hl-border);
    border-radius: 10px;
    background: #ffffff;
    scroll-behavior: smooth;
    min-height: 0;
}
.hl-scroll-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border: 1px solid var(--hl-border);
    border-radius: 6px;
    background: #ffffff;
    color: #2563eb;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
}
.hl-scroll-btn:hover { background: #eff6ff; border-color: #bfdbfe; }
.hl-control {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff;
    color: #111827;
    cursor: pointer;
    font-size: 11px;
    font-weight: 700;
    transition: .15s ease;
}
.hl-control:hover { border-color: #60a5fa; background: #eff6ff; color: #1d4ed8; }
.hl-control.active { background: #eff6ff; border-color: #2563eb; color: #2563eb; }
.hl-main-action {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border: 0;
    border-radius: 8px;
    background: linear-gradient(135deg, #f97316, #ea580c);
    color: #fff;
    cursor: pointer;
    font-size: 11px;
    font-weight: 800;
    transition: opacity 0.2s;
}
.hl-main-action:hover { opacity: 0.9; }
.hl-backend-hidden {
    position: fixed !important;
    left: -10000px !important;
    top: -10000px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
.hl-transcript-side .hl-message-user { align-self: flex-end; }
.hl-transcript-side .hl-message-agent { align-self: flex-start; }
.hl-transcript-side a { text-decoration: none; }
@media (max-width: 768px) {
    body { overflow-y: auto !important; }
    .gradio-container { height: auto !important; min-height: 100vh !important; }
    .hl-shell { min-height: 100vh; height: auto; padding: 10px 12px; }
    .hl-workspace-split { grid-template-columns: 1fr; overflow: visible; }
    .hl-visualizer { min-height: 360px; border-right: 0; border-bottom: 1px solid var(--hl-border); }
    .hl-transcript-side { min-height: 360px; }
    .hl-app-header { height: auto; min-height: 48px; }
    .hl-app-header > div:last-child { gap: 6px; }
}
"""


UI_HTML = r"""
<div class="hl-shell">
    <header class="hl-app-header">
        <div class="flex items-center gap-3">
            <div class="hl-logo"><div class="hl-logo-inner">✦</div></div>
            <div class="flex flex-col">
                <div class="flex items-center gap-2">
                    <span class="text-sm font-black text-gray-900">HamzaLive</span>
                    <span class="px-1.5 py-0.5 border border-purple-300 rounded-full bg-purple-50 text-purple-700 text-[9px] font-extrabold uppercase">PRO 2.0</span>
                </div>
                <span class="text-[10px] text-gray-500">Ultra-low Latency Voice & Vision Multimodal AI</span>
            </div>
        </div>
        <div class="flex items-center gap-2.5">
            <div class="flex items-center gap-2 px-3 py-1 border border-gray-200 rounded-full bg-slate-50 text-xs font-semibold text-gray-900" id="hl-status-badge">
                <span class="w-2 h-2 rounded-full hl-status-dot live"></span>
                <span id="hl-status-text" class="text-[11px]">Live Room Active</span>
            </div>
            <button class="w-8 h-8 grid place-items-center border border-gray-200 rounded-lg bg-white hover:bg-gray-50 text-gray-700 font-bold transition" title="Settings">⚙</button>
        </div>
    </header>

    <main class="flex-1 flex flex-col gap-2.5 min-h-0">
        <div class="hl-panel">
            <div class="flex items-center justify-between gap-3 flex-wrap px-4 py-2.5 border-b border-gray-200 bg-white">
                <div class="flex items-center gap-2 flex-wrap">
                    <div class="flex items-center gap-1.5 px-2.5 py-1 border border-blue-200 rounded-full bg-blue-50 text-blue-900 text-[10px] font-semibold">
                        <span class="text-blue-600">◆</span><span>Gemini 2.0 Flash Voice</span>
                    </div>
                    <div class="flex items-center gap-1.5 px-2.5 py-1 border border-orange-200 rounded-full bg-orange-50 text-orange-950 text-[10px] font-semibold">
                        <span class="text-orange-600"></span><span>Voice: Aoede (Warm & Conversational)</span>
                    </div>
                </div>
                <div class="flex items-center gap-1.5 text-xs text-gray-600">
                    <span class="text-amber-500 text-sm">⚡</span>
                    <span class="text-[11px]">Latency: <strong class="text-gray-900">Live</strong></span>
                </div>
            </div>

            <div class="hl-workspace-split">
                <div class="hl-visualizer">
                    <div class="hl-ambient-glow"></div>
                    <div class="relative z-10 flex flex-col items-center justify-center my-auto">
                        <canvas id="gemini-canvas" width="200" height="200"></canvas>
                    </div>
                    <div class="text-center max-w-sm z-10">
                        <div class="text-gray-900 text-base font-bold">HamzaLive Listening...</div>
                        <div class="text-gray-500 text-[11px] mt-0.5">Speak naturally into your microphone or toggle vision stream.</div>
                    </div>
                    <div class="relative z-10 flex items-center gap-1.5 px-2.5 py-1 border border-gray-200 rounded-xl bg-white text-[10px] text-gray-800">
                        <span class="w-1.5 h-1.5 rounded-full bg-green-600"></span><span>VAD Auto-detection Enabled</span>
                    </div>
                </div>

                <div class="hl-transcript-side">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-1.5 text-gray-900 text-[10px] font-black tracking-wider uppercase"><span>LIVE TRANSCRIPT</span></div>
                        <button class="hl-scroll-btn" onclick="scrollToBottomTranscript()"><span>Scroll to Bottom</span> ↓</button>
                    </div>

                    <div class="hl-transcript-feed" id="hl-transcript-feed">
                        <div class="flex flex-col max-w-[85%] gap-0.5 self-start items-start">
                            <div class="text-[9px] text-gray-500">HamzaLive · 08:12 PM</div>
                            <div class="px-3 py-2 border border-gray-200 rounded-xl bg-gray-100 text-gray-900 text-[11px] leading-relaxed">Hi there! How is your day going so far?</div>
                        </div>
                        <div class="flex flex-col max-w-[85%] gap-0.5 self-end items-end">
                            <div class="text-[9px] text-gray-500">You · 08:12 PM</div>
                            <div class="px-3 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white text-[11px] leading-relaxed">Introduce yourself in full</div>
                        </div>
                        <div class="flex flex-col max-w-[85%] gap-0.5 self-start items-start">
                            <div class="text-[9px] text-gray-500">HamzaLive · 08:12 PM</div>
                            <div class="px-3 py-2 border border-gray-200 rounded-xl bg-gray-100 text-gray-900 text-[11px] leading-relaxed">I am HamzaLive, your ultra-low latency voice and vision multimodal assistant powered by Gemini 2.0 Flash and LiveKit WebSockets!</div>
                        </div>
                    </div>

                    <form onsubmit="handleSend(event)" class="relative flex items-center mt-1">
                        <input type="text" id="transcript-input" class="w-full py-2 pl-3 pr-10 border border-gray-200 rounded-lg outline-none bg-white text-gray-900 text-[11px] focus:border-blue-600" placeholder="Type a message or speak..." />
                        <button type="submit" class="absolute right-1 w-6 h-6 grid place-items-center border-0 rounded-md bg-blue-600 text-white hover:bg-blue-700 cursor-pointer text-xs">➔</button>
                    </form>
                </div>
            </div>

            <div class="flex items-center justify-between gap-2 px-4 py-2.5 border-t border-gray-200 bg-white">
                <div class="flex items-center gap-2">
                    <button class="hl-control" id="mic-btn" onclick="toggleControl('mic-btn', '🔇 Mute Mic', '🎙 Mic Active')"><span>🔇 Mute Mic</span></button>
                    <button class="hl-control" id="cam-btn" onclick="toggleControl('cam-btn', '💻 Vision Stream', '💻 Vision Active')"><span>💻 Vision Stream</span></button>
                    <button class="hl-control" id="spk-btn" onclick="toggleControl('spk-btn', '🔊 Speaker On', '🔇 Speaker Muted')"><span>🔊 Speaker On</span></button>
                </div>
                <button class="hl-main-action" id="session-btn" onclick="toggleSession(this)"><span>⏹ End Session</span></button>
            </div>
        </div>
    </main>

    <footer class="flex items-center justify-between py-1 text-[10px] text-gray-500 border-t border-gray-200">
        <div id="footer-status">Ready · voice-agent-411c19001dc9 · Token issued for Hamza. Backend worker is online.</div>
        <div class="flex gap-2.5 text-gray-500">
            <a href="#" class="hover:text-blue-600 text-decoration-none">Runs</a>
            <a href="#" class="hover:text-blue-600 text-decoration-none">Use via API</a>
            <a href="#" class="hover:text-blue-600 text-decoration-none">Built with Gradio</a>
            <a href="#" class="hover:text-blue-600 text-decoration-none">Settings</a>
        </div>
    </footer>
</div>
"""


UI_JS = r"""
() => {
    if (!document.getElementById("hl-tailwind-cdn")) {
        const script = document.createElement("script");
        script.id = "hl-tailwind-cdn";
        script.src = "https://cdn.tailwindcss.com";
        document.head.appendChild(script);
    }

    window.scrollToBottomTranscript = function() {
        const feed = document.getElementById("hl-transcript-feed");
        if (feed) feed.scrollTo({ top: feed.scrollHeight, behavior: "smooth" });
    };

    window.toggleControl = function(id, textOff, textOn) {
        const btn = document.getElementById(id);
        if (!btn) return;
        const label = btn.querySelector("span");
        const active = btn.classList.toggle("active");
        if (label) label.innerText = active ? textOn : textOff;
    };

    window.toggleSession = function(btn) {
        const badge = document.getElementById("hl-status-badge");
        const statusText = document.getElementById("hl-status-text");
        const footerStatus = document.getElementById("footer-status");
        const connected = btn.classList.toggle("connected");
        if (connected) {
            btn.style.background = "linear-gradient(135deg, #f97316, #ea580c)";
            btn.innerHTML = "<span>⏹ End Session</span>";
            badge.className = "flex items-center gap-2 px-3 py-1 border border-gray-200 rounded-full bg-blue-50 text-xs font-semibold text-gray-900";
            statusText.innerText = "Live Room Active";
            footerStatus.innerText = "Ready · voice-agent-411c19001dc9 · Token issued for Hamza. Backend worker is online.";
        } else {
            btn.style.background = "#2563eb";
            btn.innerHTML = "<span>▶ Connect Session</span>";
            badge.className = "flex items-center gap-2 px-3 py-1 border border-gray-200 rounded-full bg-slate-50 text-xs font-semibold text-gray-500";
            statusText.innerText = "Disconnected";
            footerStatus.innerText = "Disconnected · Click Connect Session to start backend worker.";
        }
    };

    function escapeHtml(value) {
        return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\"/g, "&quot;").replace(/'/g, "&#039;");
    }

    window.handleSend = function(event) {
        event.preventDefault();
        const input = document.getElementById("transcript-input");
        const feed = document.getElementById("hl-transcript-feed");
        const text = input && input.value.trim();
        if (!text || !feed) return;
        const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const userMsg = document.createElement("div");
        userMsg.className = "flex flex-col max-w-[85%] gap-0.5 self-end items-end";
        userMsg.innerHTML = `<div class="text-[9px] text-gray-500">You · ${now}</div><div class="px-3 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white text-[11px] leading-relaxed">${escapeHtml(text)}</div>`;
        feed.appendChild(userMsg);
        input.value = "";
        window.scrollToBottomTranscript();
        window.setTimeout(() => {
            const agentMsg = document.createElement("div");
            agentMsg.className = "flex flex-col max-w-[85%] gap-0.5 self-start items-start";
            agentMsg.innerHTML = `<div class="text-[9px] text-gray-500">HamzaLive · ${now}</div><div class="px-3 py-2 border border-gray-200 rounded-xl bg-gray-100 text-gray-900 text-[11px] leading-relaxed">Received: "${escapeHtml(text)}". I am processing your multimodal audio stream in real-time.</div>`;
            feed.appendChild(agentMsg);
            window.scrollToBottomTranscript();
        }, 800);
    };

    function initOrbVisualizer() {
        const canvas = document.getElementById("gemini-canvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        let step = 0;
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const radius = 55 + Math.sin(step * 1.5) * 8 + Math.cos(step * 0.8) * 4;
            const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, radius + 25);
            grad.addColorStop(0, "rgba(37, 99, 235, 0.85)");
            grad.addColorStop(0.4, "rgba(147, 51, 234, 0.65)");
            grad.addColorStop(0.7, "rgba(6, 182, 212, 0.35)");
            grad.addColorStop(1, "rgba(6, 182, 212, 0)");
            ctx.beginPath();
            ctx.arc(cx, cy, radius + 10, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(37, 99, 235, 0.08)";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();
            step += 0.04;
            requestAnimationFrame(animate);
        }
        animate();
    }

    window.setTimeout(() => {
        initOrbVisualizer();
        window.scrollToBottomTranscript();
    }, 500);
}
"""


LIVEKIT_UI_JS = r"""
() => {
    if (!document.getElementById("hl-tailwind-cdn")) {
        const script = document.createElement("script");
        script.id = "hl-tailwind-cdn";
        script.src = "https://cdn.tailwindcss.com";
        document.head.appendChild(script);
    }

    const state = {
        room: null,
        client: null,
        loadingClient: null,
        connecting: false,
        observedToken: "",
        microphoneEnabled: false,
        cameraEnabled: false,
        speakerMuted: false,
    };

    const get = (id) => document.getElementById(id);
    const buttonFor = (id) => {
        const root = get(id);
        if (!root) return null;
        return root.matches("button") ? root : root.querySelector("button");
    };
    const clickBackend = (id) => buttonFor(id)?.click();
    const valueOf = (id) => {
        const root = get(id);
        if (!root) return "";
        const input = root.matches("input, textarea") ? root : root.querySelector("input, textarea");
        return input?.value || "";
    };
    const footer = (message) => {
        const node = get("footer-status");
        if (node) node.innerText = message;
    };
    const setStatus = (connected, message) => {
        const badge = get("hl-status-badge");
        const dot = badge?.querySelector(".hl-status-dot");
        const text = get("hl-status-text");
        if (badge) badge.className = connected
            ? "flex items-center gap-2 px-3 py-1 border border-gray-200 rounded-full bg-blue-50 text-xs font-semibold text-gray-900"
            : "flex items-center gap-2 px-3 py-1 border border-gray-200 rounded-full bg-slate-50 text-xs font-semibold text-gray-500";
        if (dot) dot.className = connected ? "w-2 h-2 rounded-full hl-status-dot live" : "w-2 h-2 rounded-full bg-gray-400";
        if (text) text.innerText = connected ? "Live Room Active" : "Disconnected";
        if (message) footer(message);
    };
    const setControl = (id, label, active) => {
        const button = get(id);
        if (!button) return;
        button.classList.toggle("active", Boolean(active));
        const span = button.querySelector("span");
        if (span) span.innerText = label;
    };

    window.scrollToBottomTranscript = function() {
        const feed = get("hl-transcript-feed");
        if (feed) feed.scrollTo({ top: feed.scrollHeight, behavior: "smooth" });
    };

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function appendTranscript(speaker, message, isUser = false) {
        const feed = get("hl-transcript-feed");
        if (!feed || !String(message || "").trim()) return;
        const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const row = document.createElement("div");
        row.className = isUser
            ? "flex flex-col max-w-[85%] gap-0.5 self-end items-end"
            : "flex flex-col max-w-[85%] gap-0.5 self-start items-start";
        const bubble = isUser
            ? "px-3 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white text-[11px] leading-relaxed"
            : "px-3 py-2 border border-gray-200 rounded-xl bg-gray-100 text-gray-900 text-[11px] leading-relaxed";
        row.innerHTML = `<div class="text-[9px] text-gray-500">${escapeHtml(speaker)} · ${time}</div><div class="${bubble}">${escapeHtml(message)}</div>`;
        feed.appendChild(row);
        window.scrollToBottomTranscript();
    }

    async function loadLiveKit() {
        if (window.LivekitClient || window.LiveKitClient) return window.LivekitClient || window.LiveKitClient;
        if (state.loadingClient) return state.loadingClient;
        state.loadingClient = new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = "https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js";
            script.async = true;
            script.onload = () => resolve(window.LivekitClient || window.LiveKitClient);
            script.onerror = () => reject(new Error("Could not load the LiveKit browser client."));
            document.head.appendChild(script);
        });
        return state.loadingClient;
    }

    function attachTranscriptionHandlers(room) {
        const handleText = async (reader, participantInfo) => {
            try {
                const message = await reader.readAll();
                const attributes = reader.info?.attributes || {};
                const isUser = participantInfo?.identity === room.localParticipant.identity;
                appendTranscript(isUser ? "You" : "HamzaLive", message, isUser);
            } catch (error) {
                footer("Transcript error · " + (error?.message || "could not read transcript"));
            }
        };
        if (room.registerTextStreamHandler) {
            room.registerTextStreamHandler("lk.transcription", handleText);
        } else {
            room.on("transcriptionReceived", (segments, participant) => {
                (segments || []).forEach((segment) => {
                    const isUser = participant?.identity === room.localParticipant.identity;
                    appendTranscript(isUser ? "You" : "HamzaLive", segment.text, isUser);
                });
            });
        }
    }

    async function disconnectRoom(requestBackend = false) {
        const room = state.room;
        state.room = null;
        state.connecting = false;
        state.microphoneEnabled = false;
        state.cameraEnabled = false;
        if (room) {
            try { await room.disconnect(); } catch (_) {}
        }
        const button = get("session-btn");
        if (button) {
            button.classList.remove("connected");
            button.style.background = "#2563eb";
            button.innerHTML = "<span>▶ Connect Session</span>";
        }
        setControl("mic-btn", "🔇 Mute Mic", false);
        setControl("cam-btn", "💻 Vision Stream", false);
        setControl("spk-btn", "🔊 Speaker On", false);
        setStatus(false, "Disconnected · Click Connect Session to start the backend worker.");
        if (requestBackend) clickBackend("hl-end-backend");
    }

    async function connectRoom(url, token) {
        if (state.room || state.connecting) return;
        state.connecting = true;
        setStatus(false, "Connecting to LiveKit…");
        try {
            const client = await loadLiveKit();
            if (!client) throw new Error("LiveKit browser client is unavailable.");
            const room = new client.Room({ adaptiveStream: true, dynacast: true });
            state.room = room;
            room.on("trackSubscribed", (track) => {
                if (track.kind === "audio") {
                    const element = track.attach();
                    element.autoplay = true;
                    element.muted = state.speakerMuted;
                    document.body.appendChild(element);
                }
            });
            room.on("trackUnsubscribed", (track) => track.detach().forEach((element) => element.remove()));
            room.on("disconnected", () => {
                if (state.room === room) disconnectRoom(false);
            });
            attachTranscriptionHandlers(room);
            await room.connect(url, token);
            await room.localParticipant.setMicrophoneEnabled(false);
            state.connecting = false;
            const button = get("session-btn");
            if (button) {
                button.classList.add("connected");
                button.style.background = "linear-gradient(135deg, #f97316, #ea580c)";
                button.innerHTML = "<span>⏹ End Session</span>";
            }
            setControl("mic-btn", "🎙 Unmute Mic", false);
            setStatus(true, "Connected · microphone is muted until you enable it.");
        } catch (error) {
            state.room = null;
            state.connecting = false;
            setStatus(false, "LiveKit connection failed · " + (error?.message || "check your credentials"));
        }
    }

    async function toggleMicrophone() {
        if (!state.room) { footer("Connect a LiveKit session first."); return; }
        try {
            await state.room.startAudio().catch(() => {});
            state.microphoneEnabled = !state.microphoneEnabled;
            await state.room.localParticipant.setMicrophoneEnabled(state.microphoneEnabled);
            setControl("mic-btn", state.microphoneEnabled ? "🔊 Mic Active" : "🎙 Unmute Mic", state.microphoneEnabled);
        } catch (error) {
            footer("Microphone error · " + (error?.message || "permission was not granted"));
        }
    }

    async function toggleCamera() {
        if (!state.room) { footer("Connect a LiveKit session first."); return; }
        try {
            state.cameraEnabled = !state.cameraEnabled;
            await state.room.localParticipant.setCameraEnabled(state.cameraEnabled);
            setControl("cam-btn", state.cameraEnabled ? "💻 Vision Active" : "💻 Vision Stream", state.cameraEnabled);
        } catch (error) {
            state.cameraEnabled = false;
            footer("Camera error · " + (error?.message || "permission was not granted"));
        }
    }

    function toggleSpeaker() {
        if (!state.room) { footer("Connect a LiveKit session first."); return; }
        state.speakerMuted = !state.speakerMuted;
        document.querySelectorAll("audio, video").forEach((element) => { element.muted = state.speakerMuted; });
        setControl("spk-btn", state.speakerMuted ? "🔇 Speaker Muted" : "🔊 Speaker On", state.speakerMuted);
    }

    window.toggleControl = function(id) {
        if (id === "mic-btn") return toggleMicrophone();
        if (id === "cam-btn") return toggleCamera();
        if (id === "spk-btn") return toggleSpeaker();
    };

    window.toggleSession = function() {
        if (state.room || state.connecting) {
            disconnectRoom(true);
            return;
        }
        footer("Requesting a LiveKit token…");
        clickBackend("hl-start-backend");
    };

    window.handleSend = async function(event) {
        event.preventDefault();
        const input = get("transcript-input");
        const text = input?.value.trim();
        if (!text) return;
        if (!state.room) { footer("Connect a LiveKit session first."); return; }
        try {
            await state.room.localParticipant.sendText(text, { topic: "lk.chat" });
            appendTranscript("You", text, true);
            input.value = "";
        } catch (error) {
            footer("Message error · " + (error?.message || "could not send message"));
        }
    };

    function syncToken() {
        const token = valueOf("hl-token-state");
        const url = valueOf("hl-url-state");
        if (token && token !== state.observedToken) {
            state.observedToken = token;
            connectRoom(url, token);
        } else if (!token && state.observedToken) {
            state.observedToken = "";
            if (state.room) disconnectRoom(false);
        }
    }

    function initOrbVisualizer() {
        const canvas = document.getElementById("gemini-canvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        let step = 0;
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const radius = 55 + Math.sin(step * 1.5) * 8 + Math.cos(step * 0.8) * 4;
            const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, radius + 25);
            grad.addColorStop(0, "rgba(37, 99, 235, 0.85)");
            grad.addColorStop(0.4, "rgba(147, 51, 234, 0.65)");
            grad.addColorStop(0.7, "rgba(6, 182, 212, 0.35)");
            grad.addColorStop(1, "rgba(6, 182, 212, 0)");
            ctx.beginPath(); ctx.arc(cx, cy, radius + 10, 0, Math.PI * 2); ctx.fillStyle = "rgba(37, 99, 235, 0.08)"; ctx.fill();
            ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.fillStyle = grad; ctx.fill();
            step += 0.04;
            requestAnimationFrame(animate);
        }
        animate();
    }

    window.setTimeout(() => {
        initOrbVisualizer();
        window.scrollToBottomTranscript();
        window.setInterval(syncToken, 700);
    }, 500);
}
"""


with gr.Blocks(title="HamzaLive") as demo:
    token_state = gr.Textbox(value="", show_label=False, container=False, elem_id="hl-token-state", elem_classes=["hl-backend-hidden"])
    url_state = gr.Textbox(value=os.getenv("LIVEKIT_URL", ""), show_label=False, container=False, elem_id="hl-url-state", elem_classes=["hl-backend-hidden"])
    room_input = gr.Textbox(value=DEFAULT_ROOM, show_label=False, container=False, elem_id="hl-room-input", elem_classes=["hl-backend-hidden"])
    identity_input = gr.Textbox(value=DEFAULT_IDENTITY, show_label=False, container=False, elem_id="hl-identity-input", elem_classes=["hl-backend-hidden"])
    connection_state = gr.Textbox(value="Ready to connect", show_label=False, container=False, elem_id="hl-connection-state", elem_classes=["hl-backend-hidden"])
    connection_hint = gr.Textbox(value="", show_label=False, container=False, elem_id="hl-connection-hint", elem_classes=["hl-backend-hidden"])
    start_backend = gr.Button("Start backend", elem_id="hl-start-backend", elem_classes=["hl-backend-hidden"])
    end_backend = gr.Button("End backend", elem_id="hl-end-backend", elem_classes=["hl-backend-hidden"])
    gr.HTML(UI_HTML)

    start_backend.click(
        fn=start_session,
        inputs=[room_input, identity_input],
        outputs=[token_state, url_state, connection_state, connection_hint],
    )
    end_backend.click(
        fn=disconnect_session,
        inputs=[],
        outputs=[token_state, url_state, connection_state, connection_hint],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS, js=LIVEKIT_UI_JS)
