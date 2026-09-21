"""HamzaLive Gradio workspace for the LiveKit voice agent.

Run with:

    python app.py

The server-side functions in this module create room tokens and manage the
LiveKit worker from agent.py. The HamzaLive interface below runs the live
browser controls and transcript through the LiveKit JavaScript client.
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
LIVEKIT_CLIENT_CDN = "https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js"

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
    """Start the LiveKit worker defined by agent.py."""

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
    """Stop only the worker process started by this dashboard."""

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
    stop_worker()
    return "", "", "Disconnected", "The browser session ended and the backend worker is offline."


def _shutdown_worker() -> None:
    if _worker_is_running():
        stop_worker()


atexit.register(_shutdown_worker)


APP_CSS = """
:root {
  --hl-bg: #080c14;
  --hl-surface: #131927;
  --hl-surface-2: #0d1422;
  --hl-border: #212b3e;
  --hl-muted: #94a3b8;
  --hl-text: #f3f4f6;
  --hl-blue: #4285f4;
  --hl-purple: #a142f4;
  --hl-cyan: #00e5ff;
  --hl-shadow: 0 24px 60px rgba(0, 0, 0, .28);
}
* { box-sizing: border-box; }
body, .gradio-container {
  background: var(--hl-bg) !important;
  color: var(--hl-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}
.gradio-container {
  max-width: 1440px !important;
  min-height: 100vh;
  padding: 0 24px 30px !important;
}
.gradio-container > .main { padding-top: 0 !important; }
.hl-app-header {
  width: 100vw;
  margin-left: calc(50% - 50vw);
  height: 66px;
  padding: 0 max(24px, calc((100vw - 1240px) / 2));
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  background: rgba(19, 25, 39, .78);
  border-bottom: 1px solid rgba(255, 255, 255, .08);
  backdrop-filter: blur(16px);
}
.hl-brand, .hl-brand-line, .hl-header-actions, .hl-status-badge, .hl-action-row,
.hl-pill, .hl-latency, .hl-transcript-title, .hl-transcript-actions,
.hl-control-group, .hl-control, .hl-main-action, .hl-mode-indicator {
  display: flex;
  align-items: center;
}
.hl-brand { gap: 11px; }
.hl-logo {
  width: 38px; height: 38px; display: grid; place-items: center; padding: 2px;
  border-radius: 12px; background: linear-gradient(135deg, #2563eb, #9333ea, #22d3ee);
  box-shadow: 0 10px 24px rgba(161, 66, 244, .22);
}
.hl-logo-inner {
  width: 100%; height: 100%; display: grid; place-items: center; border-radius: 10px;
  background: #020617; color: var(--hl-cyan); font-size: 18px; font-weight: 800;
}
.hl-brand-copy { display: block; }
.hl-brand-line { gap: 8px; }
.hl-brand-name { color: var(--hl-text); font-size: 15px; font-weight: 900; letter-spacing: -.02em; }
.hl-gradient-text {
  background: linear-gradient(135deg, #4285f4, #a142f4 50%, #00e5ff);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hl-pro {
  padding: 3px 7px; border: 1px solid rgba(161, 66, 244, .25); border-radius: 999px;
  background: rgba(161, 66, 244, .1); color: #c084fc; font-size: 9px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase;
}
.hl-tagline { margin-top: 3px; color: #94a3b8; font-size: 10px; }
.hl-header-actions { gap: 10px; }
.hl-status-badge {
  gap: 8px; padding: 7px 11px; border: 1px solid #1e293b; border-radius: 999px;
  background: rgba(2, 6, 23, .72); color: #94a3b8; font-size: 10px; font-weight: 700;
  transition: .2s ease;
}
.hl-status-badge.live { border-color: rgba(0, 229, 255, .35); background: rgba(8, 47, 73, .58); color: #67e8f9; }
.hl-status-dot { width: 7px; height: 7px; border-radius: 50%; background: #64748b; }
.hl-status-dot.live { background: var(--hl-cyan); box-shadow: 0 0 0 4px rgba(0, 229, 255, .14); }
.hl-settings-button {
  width: 32px; height: 32px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,.1);
  border-radius: 10px; background: rgba(30, 41, 59, .6); color: #cbd5e1; cursor: pointer; font-size: 15px;
}
.hl-settings-button:hover { background: #1e293b; color: #fff; }
.hl-shell { max-width: 1240px; margin: 0 auto; }
.hl-main { display: flex; flex-direction: column; gap: 20px; padding-top: 24px; }
.hl-panel {
  position: relative; overflow: hidden; min-height: 610px; display: flex; flex-direction: column;
  border: 1px solid rgba(255,255,255,.08); border-radius: 25px; background: rgba(19,25,39,.7);
  box-shadow: var(--hl-shadow); backdrop-filter: blur(16px);
}
.hl-action-row {
  justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 16px 22px;
  border-bottom: 1px solid rgba(148,163,184,.12); background: rgba(2,6,23,.42);
}
.hl-action-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.hl-pill {
  gap: 7px; padding: 6px 10px; border: 1px solid #1e293b; border-radius: 999px;
  background: rgba(2,6,23,.72); color: #cbd5e1; font-size: 10px; font-weight: 600;
}
.hl-pill-icon { color: var(--hl-cyan); font-size: 12px; }
.hl-pill.voice .hl-pill-icon { color: #c084fc; }
.hl-latency { gap: 6px; color: #94a3b8; font-size: 10px; }
.hl-latency-icon { color: #fbbf24; font-size: 14px; }
.hl-latency strong { color: #e2e8f0; }
.hl-workspace-grid { flex: 1; min-height: 0; gap: 0 !important; align-items: stretch !important; }
.hl-visual-column, .hl-transcript-column { min-width: 0 !important; padding: 0 !important; }
.hl-visual-column { border-right: 1px solid rgba(148,163,184,.12); }
.hl-visualizer {
  position: relative; min-height: 475px; display: flex; flex-direction: column; align-items: center; justify-content: space-between;
  overflow: hidden; padding: 24px 28px 20px; background: linear-gradient(180deg, #020617, rgba(15,23,42,.7), #020617);
}
.hl-visualizer::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(circle at center, rgba(66,133,244,.2), rgba(161,66,244,.1) 40%, transparent 70%);
  opacity: .55;
}
.hl-ambient-glow { position: absolute; inset: 0; pointer-events: none; background: radial-gradient(circle at center, rgba(0,229,255,.17), transparent 62%); opacity: .35; transition: .6s ease; }
.hl-camera-feed { position: absolute; inset: 0; z-index: 10; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #020617; }
.hl-camera-feed.hidden, .hl-hidden { display: none !important; }
.hl-camera-feed video { width: 100%; height: 100%; object-fit: cover; opacity: .64; transform: scaleX(-1); }
.hl-camera-label { position: absolute; top: 15px; left: 15px; display: flex; align-items: center; gap: 7px; padding: 7px 10px; border: 1px solid #334155; border-radius: 999px; background: rgba(2,6,23,.82); color: #67e8f9; font-size: 10px; }
.hl-camera-label span { width: 7px; height: 7px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,.15); }
.hl-camera-frame { position: absolute; inset: 15px; border: 2px solid rgba(34,211,238,.2); border-radius: 14px; pointer-events: none; }
.hl-orb-area { position: relative; z-index: 20; display: flex; flex-direction: column; align-items: center; justify-content: center; margin: auto; padding: 20px 0; }
#gemini-canvas { width: 280px; height: 280px; border-radius: 50%; cursor: pointer; filter: drop-shadow(0 20px 45px rgba(66,133,244,.18)); transition: transform .4s ease; }
#gemini-canvas:hover { transform: scale(1.04); }
.hl-orb-core {
  position: absolute; width: 112px; height: 112px; display: flex; flex-direction: column; align-items: center; justify-content: center;
  border: 1px solid rgba(100,116,139,.6); border-radius: 50%; background: rgba(2,6,23,.82); color: #fff; cursor: pointer;
  box-shadow: inset 0 0 26px rgba(15,23,42,.8); transition: .3s ease;
}
.hl-orb-core:hover { border-color: var(--hl-cyan); box-shadow: 0 0 28px rgba(0,229,255,.16), inset 0 0 26px rgba(15,23,42,.8); }
.hl-orb-icon { color: var(--hl-cyan); font-size: 27px; line-height: 1; }
.hl-orb-subtext { margin-top: 5px; color: #94a3b8; font-size: 9px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.hl-session-copy { max-width: 390px; text-align: center; }
.hl-session-title { color: #fff; font-size: 19px; font-weight: 800; letter-spacing: -.02em; }
.hl-session-subtitle { margin-top: 5px; color: #94a3b8; font-size: 10px; line-height: 1.6; }
.hl-mode-indicator {
  position: relative; z-index: 20; gap: 8px; padding: 8px 13px; border: 1px solid #1e293b; border-radius: 14px;
  background: rgba(15,23,42,.9); color: #cbd5e1; box-shadow: 0 8px 20px rgba(0,0,0,.22); font-size: 10px;
}
.hl-mode-dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; }
.hl-transcript-column {
  display: flex; flex-direction: column; justify-content: space-between; gap: 15px; padding: 20px !important;
  background: rgba(2,6,23,.56);
}
.hl-transcript-head { display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid rgba(148,163,184,.12); }
.hl-transcript-title { gap: 8px; color: #cbd5e1; font-size: 10px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
.hl-transcript-title-icon { color: #c084fc; font-size: 15px; }
.hl-transcript-actions { gap: 5px; }
.hl-transcript-action { width: 27px; height: 27px; display: grid; place-items: center; border: 0; border-radius: 8px; background: transparent; color: #94a3b8; cursor: pointer; font-size: 13px; }
.hl-transcript-action:hover { background: #1e293b; color: #fff; }
.hl-transcript-feed {
  flex: 1; min-height: 260px; max-height: 390px; overflow-y: auto; display: flex; flex-direction: column; gap: 11px;
  padding: 14px; border: 1px solid rgba(148,163,184,.12); border-radius: 15px; background: rgba(15,23,42,.5);
}
.hl-empty-transcript { min-height: 220px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; color: #64748b; text-align: center; font-size: 10px; }
.hl-empty-icon { margin-bottom: 9px; color: #334155; font-size: 27px; }
.hl-message { display: flex; flex-direction: column; max-width: 90%; gap: 4px; }
.hl-message.user { align-self: flex-end; align-items: flex-end; }
.hl-message.agent, .hl-message.system { align-self: flex-start; align-items: flex-start; }
.hl-message-meta { display: flex; gap: 8px; color: #64748b; font-size: 9px; }
.hl-message-agent { color: #67e8f9; font-weight: 800; }
.hl-message-bubble { padding: 8px 11px; border: 1px solid rgba(71,85,105,.62); border-radius: 13px 13px 13px 3px; background: #1e293b; color: #f1f5f9; font-size: 10px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.hl-message.user .hl-message-bubble { border: 0; border-radius: 13px 13px 3px 13px; background: linear-gradient(135deg, #2563eb, #9333ea); color: #fff; }
.hl-message.system .hl-message-bubble { border-color: rgba(161,66,244,.25); background: rgba(88,28,135,.22); color: #d8b4fe; font-size: 9px; }
.hl-text-form { position: relative; display: flex; align-items: center; }
.hl-text-input { width: 100%; padding: 10px 39px 10px 13px; border: 1px solid #1e293b; border-radius: 11px; outline: none; background: #0f172a; color: #fff; font-size: 10px; }
.hl-text-input::placeholder { color: #64748b; }
.hl-text-input:focus { border-color: #a142f4; box-shadow: 0 0 0 3px rgba(161,66,244,.13); }
.hl-text-submit { position: absolute; right: 5px; width: 27px; height: 27px; display: grid; place-items: center; border: 0; border-radius: 8px; background: linear-gradient(135deg, #2563eb, #9333ea); color: #fff; cursor: pointer; font-size: 14px; }
.hl-event-log { min-height: 14px; overflow: hidden; color: #64748b; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
.hl-control-dock { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; padding: 15px 18px; border-top: 1px solid rgba(148,163,184,.12); background: rgba(2,6,23,.9); }
.hl-control-group { gap: 8px; flex-wrap: wrap; }
.hl-control {
  gap: 7px; padding: 9px 12px; border: 1px solid #1e293b; border-radius: 10px; background: #0f172a; color: #cbd5e1;
  cursor: pointer; font-size: 10px; font-weight: 800; transition: .18s ease;
}
.hl-control:hover:not(:disabled) { border-color: #475569; background: #1e293b; color: #fff; }
.hl-control:disabled { opacity: .4; cursor: not-allowed; }
.hl-control-icon { color: #94a3b8; font-size: 14px; }
.hl-control-icon.cyan { color: #22d3ee; }
.hl-control-icon.purple { color: #c084fc; }
.hl-main-action {
  gap: 8px; padding: 10px 17px; border: 0; border-radius: 10px; background: linear-gradient(135deg, #2563eb, #9333ea, #06b6d4);
  color: #fff; box-shadow: 0 10px 22px rgba(126,34,206,.25); cursor: pointer; font-size: 10px; font-weight: 900; transition: .18s ease;
}
.hl-main-action:hover { filter: brightness(1.12); transform: translateY(-1px); }
.hl-main-action.connected { background: #dc2626; box-shadow: 0 10px 22px rgba(220,38,38,.2); }
.hl-server-message { max-width: 1240px; margin: 9px auto 0 !important; color: #64748b !important; font-size: 9px !important; }
.hl-server-hint { max-width: 1240px; margin: 0 auto !important; color: #475569 !important; font-size: 9px !important; }
.hl-footer { max-width: 1240px; margin: 18px auto 0; padding-top: 13px; border-top: 1px solid rgba(148,163,184,.1); color: #64748b; text-align: center; font-size: 9px; }
.hl-settings-modal {
  position: fixed; inset: 0; z-index: 100; display: flex; align-items: center; justify-content: center; padding: 18px;
  background: rgba(2,6,23,.82); backdrop-filter: blur(12px);
}
.hl-settings-card { width: min(100%, 480px); padding: 22px; border: 1px solid rgba(255,255,255,.1); border-radius: 23px; background: rgba(19,25,39,.9); box-shadow: var(--hl-shadow); }
.hl-settings-head { display: flex; align-items: center; justify-content: space-between; padding-bottom: 14px; border-bottom: 1px solid rgba(148,163,184,.12); }
.hl-settings-title { display: flex; align-items: center; gap: 8px; color: #fff; font-size: 14px; font-weight: 900; }
.hl-settings-title-icon { color: #c084fc; font-size: 17px; }
.hl-close-settings { width: 27px; height: 27px; border: 0; border-radius: 8px; background: transparent; color: #94a3b8; cursor: pointer; font-size: 18px; }
.hl-close-settings:hover { background: #1e293b; color: #fff; }
.hl-setting-group { margin-top: 16px; }
.hl-setting-label { display: block; margin-bottom: 6px; color: #94a3b8; font-size: 9px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
.hl-setting-select, .hl-setting-input { width: 100%; padding: 10px 12px; border: 1px solid #1e293b; border-radius: 10px; outline: none; background: #0f172a; color: #e2e8f0; font-size: 10px; }
.hl-setting-select:focus, .hl-setting-input:focus { border-color: #a142f4; box-shadow: 0 0 0 3px rgba(161,66,244,.12); }
.hl-range-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; color: #94a3b8; font-size: 10px; }
.hl-range { width: 100%; height: 5px; accent-color: #a142f4; }
.hl-apply-settings { width: 100%; margin-top: 18px; padding: 11px; border: 0; border-radius: 10px; background: linear-gradient(135deg, #2563eb, #9333ea); color: #fff; cursor: pointer; font-size: 10px; font-weight: 900; }
.hl-internal { position: fixed !important; left: -10000px !important; top: -10000px !important; width: 1px !important; height: 1px !important; opacity: 0 !important; pointer-events: none !important; overflow: hidden !important; }
@media (max-width: 900px) {
  .gradio-container { padding: 0 12px 24px !important; }
  .hl-app-header { padding: 0 13px; }
  .hl-tagline, .hl-status-badge { display: none; }
  .hl-main { padding-top: 15px; }
  .hl-panel { min-height: auto; }
  .hl-action-row { padding: 13px 15px; }
  .hl-visual-column { border-right: 0; border-bottom: 1px solid rgba(148,163,184,.12); }
  .hl-visualizer { min-height: 410px; padding: 16px 18px; }
  #gemini-canvas { width: 240px; height: 240px; }
  .hl-orb-core { width: 96px; height: 96px; }
  .hl-transcript-column { min-height: 425px; padding: 16px !important; }
  .hl-control-dock { align-items: stretch; }
  .hl-control-group { width: 100%; }
  .hl-main-action { flex: 1 1 190px; justify-content: center; }
}

/* Light workspace theme */
body, .gradio-container {
  background: #f6f8fc !important;
  color: #0f172a !important;
}
.hl-app-header {
  background: rgba(255,255,255,.94);
  border-bottom-color: #e2e8f0;
}
.hl-brand-name { color: #0f172a; }
.hl-tagline { color: #64748b; }
.hl-status-badge { background: #f8fafc; border-color: #dbe3ef; color: #64748b; }
.hl-settings-button { background: #fff; border-color: #dbe3ef; color: #475569; box-shadow: 0 4px 10px rgba(15,23,42,.05); }
.hl-settings-button:hover { background: #eff6ff; border-color: #93c5fd; color: #1d4ed8; }
.hl-panel {
  background: #fff;
  border-color: #dbe3ef;
  box-shadow: 0 20px 50px rgba(15,23,42,.09);
}
.hl-action-row { background: #fff; border-bottom-color: #e2e8f0; }
.hl-pill { background: #eff6ff; border-color: #bfdbfe; color: #1e3a8a; }
.hl-pill.voice { background: #fff7ed; border-color: #fed7aa; color: #9a3412; }
.hl-pill-icon { color: #2563eb; }
.hl-pill.voice .hl-pill-icon { color: #ea580c; }
.hl-visual-column { border-right-color: #e2e8f0; }
.hl-visualizer {
  background: radial-gradient(circle at 50% 25%, rgba(219,234,254,.9), transparent 42%), linear-gradient(180deg, #ffffff, #eff6ff);
}
.hl-visualizer::before { background: radial-gradient(circle at center, rgba(59,130,246,.16), rgba(249,115,22,.08) 42%, transparent 72%); }
.hl-ambient-glow { background: radial-gradient(circle at center, rgba(37,99,235,.17), rgba(249,115,22,.08), transparent 68%); }
.hl-orb-core { border-color: #bfdbfe; background: rgba(255,255,255,.9); color: #1d4ed8; box-shadow: 0 8px 22px rgba(37,99,235,.12), inset 0 0 20px rgba(219,234,254,.7); }
.hl-orb-core:hover { border-color: #2563eb; box-shadow: 0 0 28px rgba(37,99,235,.18), inset 0 0 20px rgba(219,234,254,.7); }
.hl-orb-icon { color: #2563eb; }
.hl-orb-subtext { color: #64748b; }
.hl-session-title { color: #0f172a; }
.hl-session-subtitle { color: #64748b; }
.hl-mode-indicator { background: rgba(255,255,255,.9); border-color: #dbe3ef; color: #475569; box-shadow: 0 8px 20px rgba(15,23,42,.08); }
.hl-mode-dot { background: #16a34a; }
.hl-transcript-column { background: #f8fafc; border-left-color: #e2e8f0; }
.hl-transcript-head { border-bottom-color: #e2e8f0; }
.hl-transcript-title { color: #334155; }
.hl-transcript-action:hover { background: #eff6ff; color: #1d4ed8; }
.hl-transcript-feed { border-color: #dbe3ef; background: #fff; }
.hl-empty-transcript { color: #94a3b8; }
.hl-empty-icon { color: #cbd5e1; }
.hl-message-meta { color: #94a3b8; }
.hl-message-agent { color: #2563eb; }
.hl-message-bubble { border-color: #dbe3ef; background: #f1f5f9; color: #1e293b; }
.hl-message.user .hl-message-bubble { background: linear-gradient(135deg, #2563eb, #1d4ed8); }
.hl-message.system .hl-message-bubble { border-color: #fed7aa; background: #fff7ed; color: #c2410c; }
.hl-text-input { border-color: #dbe3ef; background: #fff; color: #0f172a; }
.hl-text-input::placeholder { color: #94a3b8; }
.hl-text-input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,.12); }
.hl-text-submit { background: linear-gradient(135deg, #2563eb, #1d4ed8); }
.hl-control-dock { justify-content: flex-start; border-top-color: #e2e8f0; background: #fff; }
.hl-control-group { gap: 9px; }
.hl-control, .hl-main-action {
  min-height: 40px;
  border-radius: 11px;
  box-shadow: 0 4px 10px rgba(15,23,42,.06);
}
.hl-control { border-color: #cbd5e1; background: #fff; color: #1e3a8a; }
.hl-control:hover:not(:disabled) { border-color: #93c5fd; background: #eff6ff; color: #1d4ed8; box-shadow: 0 7px 15px rgba(37,99,235,.12); }
.hl-control-icon { color: #2563eb; }
.hl-control-icon.cyan { color: #2563eb; }
.hl-control-icon.purple { color: #ea580c; }
.hl-main-action {
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
  box-shadow: 0 8px 17px rgba(37,99,235,.22);
}
.hl-main-action:hover { filter: brightness(1.08); box-shadow: 0 10px 20px rgba(37,99,235,.28); }
.hl-main-action.connected { background: linear-gradient(135deg, #f97316, #ea580c); box-shadow: 0 8px 17px rgba(249,115,22,.22); }
.hl-main-action.connected:hover { background: linear-gradient(135deg, #ea580c, #c2410c); }
.hl-server-message, .hl-server-hint { color: #64748b !important; }
.hl-settings-modal { background: rgba(15,23,42,.4); }
.hl-settings-card { border-color: #dbe3ef; background: rgba(255,255,255,.98); box-shadow: 0 24px 60px rgba(15,23,42,.18); }
.hl-settings-head { border-bottom-color: #e2e8f0; }
.hl-settings-title { color: #0f172a; }
.hl-close-settings { color: #64748b; }
.hl-close-settings:hover { background: #eff6ff; color: #1d4ed8; }
.hl-setting-label, .hl-range-row { color: #64748b; }
.hl-setting-select, .hl-setting-input { border-color: #dbe3ef; background: #f8fafc; color: #1e293b; }
.hl-setting-select:focus, .hl-setting-input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,.12); }
.hl-apply-settings { background: linear-gradient(135deg, #2563eb, #1d4ed8); box-shadow: 0 8px 16px rgba(37,99,235,.2); }
.hl-footer { border-top-color: #e2e8f0; color: #94a3b8; }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 3px solid rgba(37,99,235,.25); outline-offset: 2px; }
@media (max-width: 900px) {
  .hl-control-group { width: 100%; }
  .hl-main-action { flex: 1 1 190px; }
}

/* Final white professional treatment */
body, .gradio-container {
  background: #ffffff !important;
  color: #111827 !important;
}
.hl-app-header {
  background: #ffffff;
  border-bottom-color: #e5e7eb;
}
.hl-brand-name {
  color: #111827 !important;
  background: none !important;
  -webkit-text-fill-color: #111827 !important;
}
.hl-tagline, .hl-latency, .hl-server-message, .hl-server-hint { color: #4b5563 !important; }
.hl-status-badge { background: #f9fafb; border-color: #d1d5db; color: #111827; }
.hl-settings-button { background: #ffffff; border-color: #d1d5db; color: #111827; }
.hl-panel { background: #ffffff; border-color: #d1d5db; box-shadow: 0 18px 45px rgba(15,23,42,.08); }
.hl-action-row { background: #ffffff; border-bottom-color: #e5e7eb; }
.hl-pill { background: #eff6ff; border-color: #bfdbfe; color: #1e3a8a; }
.hl-pill.voice { background: #fff7ed; border-color: #fed7aa; color: #9a3412; }
.hl-visualizer {
  background: radial-gradient(circle at 50% 28%, #eff6ff, transparent 42%), linear-gradient(180deg, #ffffff, #f8fafc);
}
.hl-visualizer::before { background: radial-gradient(circle at center, rgba(37,99,235,.11), rgba(249,115,22,.06) 45%, transparent 72%); }
.hl-session-title { color: #111827 !important; }
.hl-session-subtitle { color: #4b5563 !important; }
.hl-mode-indicator { background: #ffffff; border-color: #d1d5db; color: #111827; }
.hl-transcript-column { background: #f9fafb; border-left-color: #e5e7eb; }
.hl-transcript-head { border-bottom-color: #e5e7eb; }
.hl-transcript-title { color: #111827; }
.hl-transcript-feed { background: #ffffff; border-color: #d1d5db; }
.hl-empty-transcript { color: #6b7280; }
.hl-message-meta { color: #6b7280; }
.hl-message-agent { color: #1d4ed8; }
.hl-message-bubble { background: #f3f4f6; border-color: #d1d5db; color: #111827; }
.hl-text-input { background: #ffffff; border-color: #d1d5db; color: #111827; }
.hl-control-dock { background: #ffffff; border-top-color: #e5e7eb; }
.hl-control { background: #ffffff; border-color: #cbd5e1; color: #111827; }
.hl-control:hover:not(:disabled) { background: #eff6ff; border-color: #60a5fa; color: #1d4ed8; }
.hl-top-session { flex: 0 0 auto; }
.hl-action-right { display: flex; align-items: center; gap: 13px; }
.hl-settings-card { background: #ffffff; border-color: #d1d5db; }
.hl-settings-title { color: #111827; }
.hl-setting-label, .hl-range-row { color: #4b5563; }
.hl-setting-select, .hl-setting-input { background: #f9fafb; border-color: #d1d5db; color: #111827; }
.hl-footer { color: #6b7280; border-top-color: #e5e7eb; }
@media (max-width: 900px) {
  .hl-action-right { width: 100%; justify-content: space-between; }
  .hl-top-session { flex: 1 1 auto; justify-content: center; }
}
"""


LIVEKIT_CLIENT_JS = """
() => {
  const CDN = "https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js";
  const state = {
    room: null,
    client: null,
    observedToken: "",
    loadingClient: null,
    microphoneEnabled: false,
    speakerMuted: false,
    cameraStream: null,
    status: "idle",
    model: "Gemini 2.0 Flash Voice",
    voice: "Aoede (Warm & Conversational)",
    userName: "Hamza",
    canvasAnimation: null,
    canvasVisible: true,
    lastCanvasFrame: 0,
    audioPhase: 0,
    transcript: []
  };
  const transcriptEntries = new Map();
  const $ = (selector) => document.querySelector(selector);
  const setText = (selector, value) => { const node = $(selector); if (node) node.textContent = value; };
  const buttonFor = (selector) => { const root = $(selector); if (!root) return null; return root.matches("button") ? root : root.querySelector("button"); };
  const clickGradio = (selector) => { const button = buttonFor(selector); if (button) { button.click(); return true; } return false; };
  const readGradioValue = (selector) => {
    const root = $(selector);
    if (!root) return "";
    const field = root.matches("input, textarea") ? root : root.querySelector("input, textarea");
    return field ? field.value : "";
  };
  const setDisabled = (selector, disabled) => { const node = $(selector); if (node) node.disabled = disabled; };

  function setHeaderStatus(connected) {
    const badge = $("#header-status-badge");
    const dot = $("#status-dot");
    const ping = $("#status-ping");
    if (badge) badge.classList.toggle("live", connected);
    if (dot) dot.classList.toggle("live", connected);
    if (ping) ping.classList.toggle("hl-hidden", !connected);
    setText("#status-text", connected ? "Live Room Active" : "Ready to Connect");
  }

  function setStatus(status, title, subtitle) {
    state.status = status;
    setText("#session-title", title);
    setText("#session-subtitle", subtitle);
    const mode = status === "listening" ? "VAD Auto-detection Enabled" : status === "speaking" ? "HamzaLive Speaking" : status === "thinking" ? "Processing Response" : status === "muted" ? "Microphone Muted" : "VAD Auto-detection Enabled";
    setText("#mode-indicator-text", mode);
    const glow = $("#ambient-glow");
    if (glow) glow.style.opacity = status === "listening" || status === "speaking" ? ".8" : ".35";
  }

  function setMainAction(connected, connecting = false) {
    const button = $("#btn-main-session");
    if (!button) return;
    button.classList.toggle("connected", connected || connecting);
    setText("#btn-main-text", connecting ? "Connecting..." : connected ? "End HamzaLive Session" : "Start HamzaLive Session");
    setText("#btn-main-icon", connected ? "■" : "✦");
  }

  function renderOrb(timestamp = 0) {
    const canvas = $("#gemini-canvas");
    if (!canvas) return;
    if (!state.canvasVisible) { state.canvasAnimation = null; return; }
    if (timestamp - state.lastCanvasFrame < 33) {
      state.canvasAnimation = window.requestAnimationFrame(renderOrb);
      return;
    }
    state.lastCanvasFrame = timestamp;
    const ctx = canvas.getContext("2d");
    const center = canvas.width / 2;
    state.audioPhase += .035;
    let radius = 78;
    let amplitude = 4;
    if (state.status === "listening") { radius = 84 + Math.sin(state.audioPhase * 2) * 5; amplitude = 11; }
    else if (state.status === "thinking") { radius = 80 + Math.cos(state.audioPhase * 4) * 4; amplitude = 8; }
    else if (state.status === "speaking") { radius = 90 + Math.sin(state.audioPhase * 5) * 11; amplitude = 18; }
    else if (state.status === "muted") { radius = 74; amplitude = 2; }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const glow = ctx.createRadialGradient(center, center, radius * .35, center, center, radius * 1.55);
    if (state.status === "listening") {
      glow.addColorStop(0, "rgba(0,229,255,.82)"); glow.addColorStop(.5, "rgba(66,133,244,.42)"); glow.addColorStop(1, "rgba(161,66,244,0)");
    } else if (state.status === "thinking") {
      glow.addColorStop(0, "rgba(161,66,244,.9)"); glow.addColorStop(.5, "rgba(251,188,5,.42)"); glow.addColorStop(1, "rgba(66,133,244,0)");
    } else if (state.status === "speaking") {
      glow.addColorStop(0, "rgba(66,133,244,.9)"); glow.addColorStop(.45, "rgba(161,66,244,.7)"); glow.addColorStop(.82, "rgba(0,229,255,.4)"); glow.addColorStop(1, "rgba(0,0,0,0)");
    } else {
      glow.addColorStop(0, "rgba(51,65,85,.6)"); glow.addColorStop(1, "rgba(15,23,42,0)");
    }
    ctx.beginPath(); ctx.arc(center, center, radius * 1.4, 0, Math.PI * 2); ctx.fillStyle = glow; ctx.fill();
    ctx.beginPath();
    const points = 120;
    for (let i = 0; i <= points; i += 1) {
      const angle = (i / points) * Math.PI * 2;
      const wave = Math.sin(angle * 6 + state.audioPhase) * amplitude + Math.cos(angle * 4 - state.audioPhase * 1.5) * (amplitude / 2);
      const currentRadius = radius + wave;
      const x = center + Math.cos(angle) * currentRadius;
      const y = center + Math.sin(angle) * currentRadius;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    const fill = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    if (state.room) { fill.addColorStop(0, "#4285f4"); fill.addColorStop(.5, "#a142f4"); fill.addColorStop(1, "#00e5ff"); }
    else { fill.addColorStop(0, "#334155"); fill.addColorStop(1, "#0f172a"); }
    ctx.fillStyle = fill; ctx.fill();
    state.canvasAnimation = window.requestAnimationFrame(renderOrb);
  }

  function appendTranscript(speaker, message, isInterim = false) {
    const clean = String(message || "").trim();
    if (!clean) return;
    const feed = $("#transcript-feed");
    if (!feed) return;
    $("#empty-transcript-msg")?.remove();
    const isAssistant = speaker === "HamzaLive" || speaker === "Agent";
    const isSystem = speaker === "System";
    const key = speaker + "-" + Date.now() + "-" + Math.random();
    state.transcript.push({ speaker, message: clean, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    const row = document.createElement("div");
    row.className = "hl-message " + (isAssistant ? "agent" : isSystem ? "system" : "user");
    const meta = document.createElement("div");
    meta.className = "hl-message-meta";
    const name = document.createElement("span");
    name.className = isAssistant ? "hl-message-agent" : "";
    name.textContent = speaker;
    const time = document.createElement("span");
    time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    meta.append(name, time);
    const bubble = document.createElement("div");
    bubble.className = "hl-message-bubble";
    bubble.textContent = clean;
    if (isInterim) bubble.style.opacity = ".62";
    row.append(meta, bubble);
    feed.appendChild(row);
    feed.scrollTop = feed.scrollHeight;
    transcriptEntries.set(key, row);
  }

  function clearTranscript() {
    state.transcript = [];
    transcriptEntries.clear();
    const feed = $("#transcript-feed");
    if (!feed) return;
    feed.innerHTML = '<div id="empty-transcript-msg" class="hl-empty-transcript"><div class="hl-empty-icon">✦</div><p>Your real-time conversation transcript will stream here seamlessly.</p></div>';
  }

  function downloadTranscript() {
    if (!state.transcript.length) return;
    const content = state.transcript.map((item) => "[" + item.time + "] " + item.speaker + ": " + item.message).join("\\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "hamzalive-transcript-" + new Date().toISOString().slice(0, 10) + ".txt";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function setMicUi() {
    setText("#btn-mic-text", state.microphoneEnabled ? "Mute Mic" : "Unmute Mic");
    setText("#btn-mic-icon", state.microphoneEnabled ? "♩" : "♩");
  }

  function setSpeakerUi() {
    setText("#btn-speaker-text", state.speakerMuted ? "Muted" : "Speaker On");
    setText("#btn-speaker-icon", state.speakerMuted ? "◌" : "◉");
  }

  function stopCamera() {
    if (state.cameraStream) state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
    const video = $("#camera-video");
    if (video) video.srcObject = null;
    $("#camera-feed-container")?.classList.add("hl-hidden");
    setText("#btn-camera-icon", "▣");
    setText("#mode-indicator-text", state.status === "listening" ? "VAD Auto-detection Enabled" : "VAD Auto-detection Enabled");
  }

  async function toggleCamera() {
    if (!state.room) return;
    if (state.cameraStream) {
      stopCamera();
      appendTranscript("System", "Vision preview disconnected.");
      return;
    }
    try {
      state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const video = $("#camera-video");
      video.srcObject = state.cameraStream;
      $("#camera-feed-container")?.classList.remove("hl-hidden");
      setText("#btn-camera-icon", "■");
      setText("#mode-indicator-text", "Vision Stream Preview Active");
      appendTranscript("System", "Multimodal vision preview connected.");
    } catch (error) {
      appendLog(error?.message || "Camera permission was not granted");
    }
  }

  function appendLog(message) {
    const stamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setText("#hl-event-log", stamp + " · " + message);
  }

  function setLatency(connected) { setText("#latency-text", connected ? "Live" : "-- ms"); }

  async function loadClient() {
    if (window.LivekitClient || window.LiveKitClient) return window.LivekitClient || window.LiveKitClient;
    if (state.loadingClient) return state.loadingClient;
    state.loadingClient = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = CDN; script.async = true;
      script.onload = () => resolve(window.LivekitClient || window.LiveKitClient);
      script.onerror = () => reject(new Error("Could not load the LiveKit browser client."));
      document.head.appendChild(script);
    });
    return state.loadingClient;
  }

  function setAllControls(disabled) {
    setDisabled("#btn-mic", disabled);
    setDisabled("#btn-camera", disabled);
    setDisabled("#btn-speaker", disabled);
  }

  async function disconnect(reason = "Disconnected", requestBackend = false) {
    const activeRoom = state.room;
    state.room = null;
    if (activeRoom) { try { await activeRoom.disconnect(); } catch (_) {} }
    stopCamera();
    state.microphoneEnabled = false;
    state.speakerMuted = false;
    setHeaderStatus(false);
    setLatency(false);
    setMicUi();
    setSpeakerUi();
    setAllControls(true);
    setStatus("idle", reason === "Session ended" ? "Session ended safely" : "HamzaLive Voice Visualizer", reason === "Session ended" ? "Start a new session whenever you’re ready." : "Use the rectangular Start Session button above to begin.");
    setMainAction(false);
    setText("#orb-subtext", "Start");
    setText("#hl-event-log", reason);
    if (requestBackend) window.setTimeout(() => clickGradio("#hl-end-session"), 0);
  }

  async function connect(url, token) {
    try {
      await disconnect("Opening LiveKit room");
      setMainAction(false, true);
      setStatus("thinking", "Connecting to HamzaLive...", "Preparing the secure WebRTC voice channel.");
      appendLog("Connecting to room");
      state.client = await loadClient();
      if (!state.client) throw new Error("LiveKit browser client is unavailable.");
      state.room = new state.client.Room({ adaptiveStream: true, dynacast: true });
      state.room.on("trackSubscribed", (track) => {
        if (track.kind === "audio") {
          const element = track.attach();
          element.autoplay = true;
          element.muted = state.speakerMuted;
          document.body.appendChild(element);
          appendLog("Agent audio connected");
        }
      });
      state.room.on("trackUnsubscribed", (track) => {
        track.detach().forEach((element) => element.remove());
      });
      state.room.on("activeSpeakersChanged", (speakers) => {
        if (speakers && speakers.length && state.microphoneEnabled) setStatus("speaking", "HamzaLive Speaking...", "Streaming the agent response back in real time.");
        else if (state.microphoneEnabled) setStatus("listening", "HamzaLive Listening...", "Speak naturally into your microphone or toggle vision stream.");
      });
      state.room.on("disconnected", () => disconnect("LiveKit room disconnected"));
      if (state.room.registerTextStreamHandler) {
        state.room.registerTextStreamHandler("lk.transcription", async (reader, participantInfo) => {
          try {
            const message = await reader.readAll();
            const attributes = reader.info?.attributes || {};
            const isUser = participantInfo?.identity === state.room.localParticipant.identity;
            const finalText = attributes["lk.transcription_final"] !== "false";
            appendTranscript(isUser ? state.userName : "HamzaLive", message, !finalText);
            if (!isUser) setStatus("speaking", "HamzaLive Speaking...", "Streaming the agent response back in real time.");
            appendLog((isUser ? "Speech recognized" : "Agent transcript") + ": " + message.slice(0, 70));
          } catch (error) { appendLog(error?.message || "Could not read the transcript"); }
        });
      } else {
        state.room.on("transcriptionReceived", (segments, participant) => {
          (segments || []).forEach((segment) => {
            const isUser = participant?.identity === state.room.localParticipant.identity;
            appendTranscript(isUser ? state.userName : "HamzaLive", segment.text, !(segment.final ?? true));
          });
        });
      }
      await state.room.connect(url, token);
      await state.room.localParticipant.setMicrophoneEnabled(false);
      state.microphoneEnabled = false;
      setAllControls(false);
      setHeaderStatus(true);
      setLatency(true);
      setMicUi();
      setSpeakerUi();
      setMainAction(true);
      setStatus("muted", "Microphone Muted", "Unmute your microphone to speak with the HamzaLive voice agent.");
      setText("#orb-subtext", "Live");
      appendLog("Connected · microphone is off");
    } catch (error) {
      state.room = null;
      setHeaderStatus(false);
      setLatency(false);
      setMainAction(false);
      setStatus("idle", "Could not connect", error?.message || "Check the LiveKit URL and token.");
      setAllControls(true);
      appendLog(error?.message || "LiveKit connection failed");
    }
  }

  async function toggleMic() {
    if (!state.room) return;
    try {
      await state.room.startAudio().catch(() => {});
      state.microphoneEnabled = !state.microphoneEnabled;
      await state.room.localParticipant.setMicrophoneEnabled(state.microphoneEnabled);
      setMicUi();
      setStatus(state.microphoneEnabled ? "listening" : "muted", state.microphoneEnabled ? "HamzaLive Listening..." : "Microphone Muted", state.microphoneEnabled ? "Speak naturally into your microphone or toggle vision stream." : "Unmute your microphone to continue.");
      appendLog(state.microphoneEnabled ? "Microphone enabled" : "Microphone muted");
    } catch (error) { appendLog(error?.message || "Microphone permission was not granted"); }
  }

  function toggleSpeaker() {
    if (!state.room) return;
    state.speakerMuted = !state.speakerMuted;
    document.querySelectorAll("audio, video").forEach((element) => { if (element !== $("#camera-video")) element.muted = state.speakerMuted; });
    setSpeakerUi();
    appendLog(state.speakerMuted ? "Speaker muted" : "Speaker unmuted");
  }

  async function sendText() {
    if (!state.room) { appendLog("Connect a room first"); return; }
    const input = $("#text-input");
    const message = input?.value.trim();
    if (!message) return;
    try {
      await state.room.localParticipant.sendText(message, { topic: "lk.chat" });
      input.value = "";
      appendTranscript(state.userName, message);
      setStatus("thinking", "HamzaLive Thinking...", "Processing your text message.");
      appendLog("Message sent");
    } catch (error) { appendLog(error?.message || "Could not send the message"); }
  }

  function toggleSession() {
    if (state.room || state.observedToken) disconnect("Session ended", true);
    else clickGradio("#hl-start-session");
  }

  function toggleConfigModal() { $("#config-modal")?.classList.toggle("hl-hidden"); }
  function updateModelSelection() {
    state.model = $("#select-model")?.value || state.model;
    setText("#display-model-name", state.model);
  }
  function updateVoiceSelection() {
    state.voice = $("#select-voice")?.value || state.voice;
    setText("#display-voice-name", "Voice: " + state.voice);
  }
  function applySettings() {
    state.userName = $("#user-display-name")?.value.trim() || "Hamza";
    toggleConfigModal();
    appendLog("Settings applied");
  }

  function wire() {
    const orb = $("#gemini-canvas");
    if (orb && orb.dataset.wired !== "1") { orb.dataset.wired = "1"; orb.addEventListener("click", toggleSession); }
    const core = $("#orb-core-btn");
    if (core && core.dataset.wired !== "1") { core.dataset.wired = "1"; core.addEventListener("click", toggleSession); }
    const main = $("#btn-main-session");
    if (main && main.dataset.wired !== "1") { main.dataset.wired = "1"; main.addEventListener("click", toggleSession); }
    const mic = $("#btn-mic");
    if (mic && mic.dataset.wired !== "1") { mic.dataset.wired = "1"; mic.addEventListener("click", toggleMic); }
    const camera = $("#btn-camera");
    if (camera && camera.dataset.wired !== "1") { camera.dataset.wired = "1"; camera.addEventListener("click", toggleCamera); }
    const speaker = $("#btn-speaker");
    if (speaker && speaker.dataset.wired !== "1") { speaker.dataset.wired = "1"; speaker.addEventListener("click", toggleSpeaker); }
    const settings = $("#settings-button");
    if (settings && settings.dataset.wired !== "1") { settings.dataset.wired = "1"; settings.addEventListener("click", toggleConfigModal); }
    const close = $("#close-settings");
    if (close && close.dataset.wired !== "1") { close.dataset.wired = "1"; close.addEventListener("click", toggleConfigModal); }
    const apply = $("#apply-settings");
    if (apply && apply.dataset.wired !== "1") { apply.dataset.wired = "1"; apply.addEventListener("click", applySettings); }
    const clear = $("#clear-transcript");
    if (clear && clear.dataset.wired !== "1") { clear.dataset.wired = "1"; clear.addEventListener("click", clearTranscript); }
    const download = $("#download-transcript");
    if (download && download.dataset.wired !== "1") { download.dataset.wired = "1"; download.addEventListener("click", downloadTranscript); }
    const form = $("#text-form");
    if (form && form.dataset.wired !== "1") { form.dataset.wired = "1"; form.addEventListener("submit", (event) => { event.preventDefault(); sendText(); }); }
  }

  function syncFromGradio() {
    const token = readGradioValue("#hl-token-state");
    const url = readGradioValue("#hl-url-state");
    if (token && token !== state.observedToken) { state.observedToken = token; connect(url, token); }
    else if (!token && state.observedToken) { state.observedToken = ""; disconnect("Session ended"); }
  }

  const boot = () => {
    wire();
    renderOrb();
    window.setInterval(syncFromGradio, 900);
    if (window.IntersectionObserver) {
      const visibilityObserver = new IntersectionObserver((entries) => {
        state.canvasVisible = Boolean(entries[0]?.isIntersecting);
        if (state.canvasVisible && !state.canvasAnimation) renderOrb();
      });
      visibilityObserver.observe($("#gemini-canvas"));
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="HamzaLive - AI Voice Assistant") as demo:
        gr.HTML(
            """
            <header class="hl-app-header">
              <div class="hl-brand">
                <div class="hl-logo"><div class="hl-logo-inner">✦</div></div>
                <div class="hl-brand-copy">
                  <div class="hl-brand-line"><span class="hl-brand-name hl-gradient-text">HamzaLive</span><span class="hl-pro">Pro 2.0</span></div>
                  <div class="hl-tagline">Ultra-low Latency Voice & Vision Multimodal AI</div>
                </div>
              </div>
              <div class="hl-header-actions">
                <div id="header-status-badge" class="hl-status-badge"><span id="status-dot" class="hl-status-dot"></span><span id="status-ping" class="hl-hidden"></span><span id="status-text">Ready to Connect</span></div>
                <button id="settings-button" class="hl-settings-button" type="button" title="Voice and model settings">☷</button>
              </div>
            </header>
            """
        )

        with gr.Column(elem_classes="hl-shell hl-main"):
            token_state = gr.Textbox(value="", show_label=False, container=False, elem_id="hl-token-state", elem_classes="hl-internal")
            url_state = gr.Textbox(value=os.getenv("LIVEKIT_URL", ""), show_label=False, container=False, elem_id="hl-url-state", elem_classes="hl-internal")
            room_input = gr.Textbox(value=DEFAULT_ROOM, show_label=False, container=False, elem_id="hl-room-input", elem_classes="hl-internal")
            identity_input = gr.Textbox(value=DEFAULT_IDENTITY, show_label=False, container=False, elem_id="hl-identity-input", elem_classes="hl-internal")

            with gr.Group(elem_classes="hl-panel"):
                gr.HTML(
                    """
                    <div class="hl-action-row">
                      <div class="hl-action-left">
                        <div class="hl-pill"><span class="hl-pill-icon">◈</span><span id="display-model-name">Gemini 2.0 Flash Voice</span></div>
                        <div class="hl-pill voice"><span class="hl-pill-icon">◉</span><span id="display-voice-name">Voice: Aoede (Warm & Conversational)</span></div>
                      </div>
                      <div class="hl-action-right">
                        <div class="hl-latency"><span class="hl-latency-icon">ϟ</span><span>Latency: <strong id="latency-text">-- ms</strong></span></div>
                        <button id="btn-main-session" class="hl-main-action hl-top-session" type="button"><span id="btn-main-icon">✦</span><span id="btn-main-text">Start HamzaLive Session</span></button>
                      </div>
                    </div>
                    """
                )
                with gr.Row(elem_classes="hl-workspace-grid"):
                    with gr.Column(scale=7, elem_classes="hl-visual-column"):
                        gr.HTML(
                            """
                            <div id="visualizer-container" class="hl-visualizer">
                              <div id="ambient-glow" class="hl-ambient-glow"></div>
                              <div id="camera-feed-container" class="hl-camera-feed hl-hidden">
                                <video id="camera-video" autoplay muted playsinline></video>
                                <div class="hl-camera-label"><span></span><span>HamzaLive Vision Stream Active</span></div>
                                <div class="hl-camera-frame"></div>
                              </div>
                              <div class="hl-orb-area">
                                <div class="hl-canvas-wrap">
                                  <canvas id="gemini-canvas" width="280" height="280"></canvas>
                                </div>
                                <div class="hl-session-copy"><div id="session-title" class="hl-session-title">HamzaLive Voice Visualizer</div><div id="session-subtitle" class="hl-session-subtitle">Use the rectangular Start Session button above to begin.</div></div>
                              </div>
                              <div class="hl-mode-indicator"><span class="hl-mode-dot"></span><span id="mode-indicator-text">VAD Auto-detection Enabled</span></div>
                            </div>
                            """
                        )
                    with gr.Column(scale=5, elem_classes="hl-transcript-column"):
                        gr.HTML(
                            """
                            <div class="hl-transcript-head">
                              <div class="hl-transcript-title"><span class="hl-transcript-title-icon">▤</span><span>Live Transcript</span></div>
                              <div class="hl-transcript-actions"><button id="download-transcript" class="hl-transcript-action" type="button" title="Export log">⇩</button><button id="clear-transcript" class="hl-transcript-action" type="button" title="Clear transcript">⌫</button></div>
                            </div>
                            <div id="transcript-feed" class="hl-transcript-feed"><div id="empty-transcript-msg" class="hl-empty-transcript"><div class="hl-empty-icon">✦</div><p>Your real-time conversation transcript will stream here seamlessly.</p></div></div>
                            <form id="text-form" class="hl-text-form"><input id="text-input" class="hl-text-input" type="text" placeholder="Type a message or topic to discuss..." aria-label="Type a message to the agent" /><button class="hl-text-submit" type="submit">↗</button></form>
                            <div id="hl-event-log" class="hl-event-log">Waiting for you to start a session.</div>
                            """
                        )
                gr.HTML(
                    """
                    <div class="hl-control-dock">
                      <div class="hl-control-group">
                        <button id="btn-mic" class="hl-control" type="button" disabled><span id="btn-mic-icon" class="hl-control-icon">♩</span><span id="btn-mic-text">Unmute Mic</span></button>
                        <button id="btn-camera" class="hl-control" type="button" disabled><span id="btn-camera-icon" class="hl-control-icon cyan">▣</span><span>Vision Stream</span></button>
                        <button id="btn-speaker" class="hl-control" type="button" disabled><span id="btn-speaker-icon" class="hl-control-icon">◉</span><span id="btn-speaker-text">Speaker On</span></button>
                      </div>
                    </div>
                    """
                )

            connection_state = gr.Markdown("**Ready to connect**", elem_classes="hl-server-message")
            connection_hint = gr.Markdown("LiveKit credentials are loaded securely from the environment.", elem_classes="hl-server-hint")
            start_session_button = gr.Button("Start HamzaLive session", elem_id="hl-start-session", elem_classes="hl-internal")
            end_session_button = gr.Button("End HamzaLive session", elem_id="hl-end-session", elem_classes="hl-internal")

            gr.HTML(
                """
                <div id="config-modal" class="hl-settings-modal hl-hidden">
                  <div class="hl-settings-card">
                    <div class="hl-settings-head"><div class="hl-settings-title"><span class="hl-settings-title-icon">☷</span><span>HamzaLive Configuration</span></div><button id="close-settings" class="hl-close-settings" type="button">×</button></div>
                    <div class="hl-setting-group"><label class="hl-setting-label" for="select-model">AI Model Pipeline</label><select id="select-model" class="hl-setting-select"><option value="Gemini 2.0 Flash Voice">Gemini 2.0 Flash (Recommended - Lowest Latency)</option><option value="Gemini 1.5 Pro Live">Gemini 1.5 Pro Live (Reasoning Engine)</option><option value="Gemini Multimodal Live">Gemini 2.0 Multimodal Vision & Audio</option></select></div>
                    <div class="hl-setting-group"><label class="hl-setting-label" for="select-voice">Synthesized Voice Persona</label><select id="select-voice" class="hl-setting-select"><option value="Aoede (Warm & Conversational)">Aoede — Warm, Expressive & Conversational</option><option value="Puck (Upbeat & Energetic)">Puck — Upbeat, Bright & Energetic</option><option value="Charon (Deep & Professional)">Charon — Deep, Clear & Professional</option><option value="Fenrir (Authoritative & Steady)">Fenrir — Authoritative & Direct</option><option value="Kore (Calm & Soothing)">Kore — Gentle & Calm</option></select></div>
                    <div class="hl-setting-group"><label class="hl-setting-label" for="user-display-name">User Name</label><input id="user-display-name" class="hl-setting-input" type="text" value="Hamza" /></div>
                    <div class="hl-setting-group"><div class="hl-range-row"><span>Voice Activity Sensitivity (VAD)</span><span id="vad-val">Medium (35ms)</span></div><input id="vad-range" class="hl-range" type="range" min="10" max="100" value="35" /></div>
                    <button id="apply-settings" class="hl-apply-settings" type="button">Apply Settings</button>
                  </div>
                </div>
                <footer class="hl-footer">Powered by HamzaLive Engine · Real-time WebRTC AI Architecture</footer>
                """
            )

        start_session_button.click(fn=start_session, inputs=[room_input, identity_input], outputs=[token_state, url_state, connection_state, connection_hint])
        end_session_button.click(fn=disconnect_session, outputs=[token_state, url_state, connection_state, connection_hint])

    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Base(primary_hue="violet", secondary_hue="blue", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]),
        css=APP_CSS,
        js=LIVEKIT_CLIENT_JS,
        show_error=True,
    )
