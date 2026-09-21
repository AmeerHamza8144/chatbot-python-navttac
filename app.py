"""Professional Gradio control panel for the LiveKit voice agent.

Run with:

    python app.py

The dashboard is intentionally separate from ``agent.py``.  It starts the
agent's LiveKit worker in a managed subprocess, creates a room token with an
explicit ``my-agent`` dispatch, and provides a small browser-side LiveKit
console for microphone and speaker interaction.
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
AGENT_NAME = "my-agent"
DEFAULT_ROOM = f"voice-agent-{uuid.uuid4().hex[:12]}"
DEFAULT_IDENTITY = "Rizwan"
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
    """Return the status card rendered in the dashboard."""

    running = _worker_is_running()
    state = "ONLINE" if running else "OFFLINE"
    state_class = "online" if running else "offline"
    pid = str(_worker_process.pid) if running and _worker_process else "—"
    url = os.getenv("LIVEKIT_URL", "Not configured")

    return f"""
<div class="worker-status-card">
  <div class="worker-status-heading">
    <span class="status-pill {state_class}"><span class="status-dot"></span>{state}</span>
    <span class="worker-process">PID {pid}</span>
  </div>
  <div class="worker-status-grid">
    <div><span>Worker</span><strong>{AGENT_NAME}</strong></div>
    <div><span>Configuration</span><strong>{_configuration_summary()}</strong></div>
    <div><span>Endpoint</span><strong>{url}</strong></div>
    <div><span>Dispatch</span><strong>Room token · enabled</strong></div>
  </div>
</div>
"""


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


def restart_worker() -> tuple[str, str]:
    """Restart the managed worker and return a single activity message."""

    stop_worker()
    return start_worker()


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


def connect_session(
    room_name: str,
    participant_identity: str,
    livekit_url: str,
) -> tuple[str, str, str, str]:
    """Validate connection settings and hand a short-lived token to the browser."""

    room_name = (room_name or "").strip()
    participant_identity = (participant_identity or "").strip()
    livekit_url = (livekit_url or os.getenv("LIVEKIT_URL", "")).strip().rstrip("/")

    if not _env_ready():
        return "", "", "Not connected", "Add the LiveKit credentials to .env before connecting."
    if not room_name or not participant_identity:
        return "", "", "Not connected", "Room name and participant identity are required."
    if not livekit_url.startswith(("ws://", "wss://", "http://", "https://")):
        return "", "", "Not connected", "LiveKit URL must start with ws://, wss://, http://, or https://."

    try:
        token = _create_token(room_name, participant_identity)
    except Exception as exc:
        return "", "", "Connection error", f"Could not create a LiveKit token: {exc}"

    worker_note = " Worker is online." if _worker_is_running() else " Start the worker before speaking."
    return token, livekit_url, f"Ready · {room_name}", f"Token issued for {participant_identity}.{worker_note}"


def disconnect_session() -> tuple[str, str, str, str]:
    return "", "", "Disconnected", "The browser session will disconnect now."


def _shutdown_worker() -> None:
    if _worker_is_running():
        stop_worker()


atexit.register(_shutdown_worker)


APP_CSS = """
:root {
  --ink: #11213b;
  --muted: #6c7b91;
  --line: #e4eaf2;
  --surface: #ffffff;
  --canvas: #f7fbff;
  --accent: #f97316;
  --accent-deep: #ea580c;
  --accent-soft: #fff2e8;
  --sky: #38bdf8;
  --sky-soft: #e8f8ff;
  --mint: #0ba986;
  --shadow: 0 18px 45px rgba(24, 49, 87, .08);
}

body, .gradio-container { background: radial-gradient(circle at 82% 0%, #e8f8ff 0, transparent 28%), var(--canvas) !important; color: var(--ink) !important; }
.gradio-container { max-width: 1440px !important; padding: 28px 34px 42px !important; }
.app-shell { max-width: 1210px; margin: 0 auto; }
.hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin: 8px 0 26px; }
.eyebrow { color: var(--accent-deep); font-size: 11px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 10px; }
.hero h1 { color: var(--ink); font-size: 38px; line-height: 1.08; letter-spacing: -.045em; margin: 0 0 10px; }
.hero p { color: var(--muted); font-size: 15px; margin: 0; max-width: 650px; }
.hero-mark { display: flex; align-items: center; gap: 10px; color: var(--ink); font-size: 13px; font-weight: 700; white-space: nowrap; }
.hero-mark .mark { display: grid; place-items: center; width: 38px; height: 38px; color: white; border-radius: 12px; background: linear-gradient(135deg, var(--accent), var(--sky)); box-shadow: 0 9px 20px rgba(249,115,22,.25); }
.surface { border: 1px solid var(--line); border-radius: 18px; background: var(--surface); box-shadow: var(--shadow); }
.session-surface { min-height: 560px; overflow: hidden; }
.session-top { display: flex; justify-content: space-between; align-items: center; padding: 24px 26px 19px; border-bottom: 1px solid var(--line); }
.session-status-group { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; justify-content: flex-end; }
.section-kicker { color: var(--muted); font-size: 11px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
.session-title { color: var(--ink); font-size: 22px; font-weight: 750; margin-top: 7px; }
.session-state { display: inline-flex; align-items: center; gap: 8px; color: #7890a8; background: #f5f7fb; border: 1px solid var(--line); border-radius: 999px; padding: 7px 11px; font-size: 12px; font-weight: 750; }
.session-state .state-dot { width: 7px; height: 7px; border-radius: 50%; background: #a7b4c4; }
.session-state.live { color: #08785e; background: #effbf7; border-color: #c8f0e5; }
.session-state.live .state-dot { background: var(--mint); box-shadow: 0 0 0 4px rgba(11,169,134,.12); }
.live-console { padding: 24px 26px 28px; border-top: 3px solid transparent; border-image: linear-gradient(90deg, var(--accent), var(--sky)) 1; }
.console-visual { display: grid; place-items: center; min-height: 228px; border: 1px solid #dff1fb; border-radius: 15px; background: radial-gradient(circle at 50% 36%, #fff0e5 0, #eefaff 38%, #fff 72%); text-align: center; }
.voice-orb { position: relative; display: grid; place-items: center; width: 100px; height: 100px; border-radius: 50%; color: var(--accent-deep); background: #fff; border: 10px solid #fff4ec; box-shadow: 0 0 0 1px #ffd2b4, 0 14px 25px rgba(249,115,22, .14); font-size: 30px; }
.voice-orb::before, .voice-orb::after { position: absolute; content: ""; inset: -21px; border: 1px solid rgba(56,189,248,.35); border-radius: 50%; }
.voice-orb::after { inset: -34px; border-color: rgba(249,115,22,.18); }
.voice-orb.speaking { animation: pulse 1.4s ease-in-out infinite; }
.console-copy { margin-top: 17px; color: var(--ink); font-size: 14px; font-weight: 700; }
.console-subcopy { margin-top: 6px; color: var(--muted); font-size: 12px; }
@keyframes pulse { 0%,100% { transform: scale(1); box-shadow: 0 0 0 1px #dae4ff, 0 14px 25px rgba(56, 88, 170, .1); } 50% { transform: scale(1.07); box-shadow: 0 0 0 12px rgba(52,103,245,.06), 0 18px 32px rgba(56, 88, 170, .16); } }
.console-actions { display: flex; gap: 10px; align-items: center; margin-top: 18px; }
.console-actions button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 10px; padding: 10px 15px; font-size: 12px; font-weight: 750; cursor: pointer; transition: .18s ease; }
.console-actions button:hover { border-color: #b9c9ef; transform: translateY(-1px); }
.console-actions button.primary { color: #fff; border-color: var(--accent); background: linear-gradient(135deg, var(--accent), var(--accent-deep)); }
.console-actions button.danger { color: #b84052; }
.data-row { display: flex; align-items: center; gap: 9px; margin-top: 19px; }
.data-row input { flex: 1; min-width: 0; border: 1px solid var(--line); border-radius: 9px; padding: 10px 12px; color: var(--ink); background: #fbfcfe; font-size: 12px; outline: none; }
.data-row input:focus { border-color: #9cb5fa; box-shadow: 0 0 0 3px rgba(52,103,245,.09); }
.data-row button { border: 0; border-radius: 9px; padding: 10px 13px; color: #fff; background: #152847; font-size: 12px; font-weight: 750; cursor: pointer; }
.event-log { margin-top: 17px; height: 29px; color: var(--muted); font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.transcript-panel { margin-top: 18px; border: 1px solid #e4ebf6; border-radius: 13px; background: #fbfcff; overflow: hidden; }
.transcript-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #e8edf5; }
.transcript-title { color: var(--ink); font-size: 12px; font-weight: 800; }
.transcript-caption { color: var(--muted); font-size: 10px; margin-top: 3px; }
.transcript-clear { border: 0; color: #58709a; background: transparent; font-size: 10px; font-weight: 750; cursor: pointer; }
.transcript-list { display: flex; flex-direction: column; gap: 9px; min-height: 73px; max-height: 190px; overflow-y: auto; padding: 12px 14px; }
.transcript-empty { display: grid; place-items: center; min-height: 48px; color: #91a0b5; font-size: 11px; text-align: center; }
.transcript-entry { display: flex; flex-direction: column; max-width: 85%; }
.transcript-entry.user { align-self: flex-end; align-items: flex-end; }
.transcript-entry.agent { align-self: flex-start; align-items: flex-start; }
.transcript-speaker { color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; margin: 0 4px 4px; }
.transcript-bubble { color: var(--ink); background: #fff; border: 1px solid #e1e8f3; border-radius: 11px 11px 11px 3px; padding: 8px 10px; font-size: 12px; line-height: 1.42; white-space: pre-wrap; word-break: break-word; }
.transcript-entry.user .transcript-bubble { color: #fff; border-color: var(--accent); background: var(--accent); border-radius: 11px 11px 3px 11px; }
.transcript-entry.interim .transcript-bubble { opacity: .68; font-style: italic; }
.call-control:disabled { opacity: .45; cursor: not-allowed !important; transform: none !important; }
.right-stack { gap: 18px; }
.sidebar-panel { order: -1; }
.main-panel { order: 1; }
.control-surface { padding: 22px; }
.control-title { color: var(--ink); font-size: 16px; font-weight: 750; margin-bottom: 5px; }
.control-description { color: var(--muted); font-size: 12px; line-height: 1.55; margin-bottom: 15px; }
.worker-status-card { padding: 2px 0; }
.worker-status-heading { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.status-pill { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 6px 10px; font-size: 11px; font-weight: 800; letter-spacing: .08em; }
.status-pill.online { color: #08785e; background: #effbf7; }
.status-pill.offline { color: #7c6870; background: #f8f3f5; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.worker-process { color: var(--muted); font-size: 11px; }
.worker-status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 13px 20px; }
.worker-status-grid div { min-width: 0; }
.worker-status-grid span { display: block; color: var(--muted); font-size: 10px; margin-bottom: 4px; }
.worker-status-grid strong { display: block; color: var(--ink); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.worker-buttons { display: flex; gap: 8px; margin-top: 19px; }
.worker-buttons button { flex: 1; }
.hint { color: var(--muted); font-size: 11px; line-height: 1.45; padding: 10px 12px; border-radius: 9px; background: #f7f9fd; }
.config-note { margin-top: 15px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; line-height: 1.55; }
.config-note code { color: #435f9e; background: #eef3ff; padding: 2px 5px; border-radius: 4px; }
.flow-surface { padding: 19px 22px; }
.flow-title { color: var(--ink); font-size: 14px; font-weight: 750; margin-bottom: 13px; }
.flow-step { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 11px; line-height: 1.4; margin-top: 10px; }
.flow-step span { display: grid; place-items: center; flex: 0 0 auto; width: 22px; height: 22px; color: var(--accent); background: var(--accent-soft); border-radius: 50%; font-size: 10px; font-weight: 800; }
.advanced-panel { border: 1px solid var(--line) !important; border-radius: 15px !important; background: rgba(255,255,255,.72) !important; box-shadow: none !important; }
.advanced-panel > .label-wrap { color: var(--ink) !important; font-size: 12px !important; font-weight: 750 !important; }
.settings-label { color: var(--muted); font-size: 11px; font-weight: 750; margin: 2px 0 5px; }
.lk-internal-state { position: fixed !important; left: -10000px !important; top: -10000px !important; width: 1px !important; height: 1px !important; opacity: 0 !important; pointer-events: none !important; overflow: hidden !important; }
footer { display: none !important; }
@media (max-width: 900px) { .gradio-container { padding: 20px 15px 30px !important; } .hero { align-items: flex-start; flex-direction: column; } .hero h1 { font-size: 31px; } .session-surface { min-height: auto; } }
"""


LIVEKIT_CLIENT_JS = f"""
() => {{
  const CDN = {LIVEKIT_CLIENT_CDN!r};
  let room = null;
  let observedToken = "";
  let loadingClient = null;
  let microphoneEnabled = false;
  let speakerMuted = false;
  const transcriptEntries = new Map();

  const $ = (selector) => document.querySelector(selector);
  const setText = (selector, text) => {{ const node = $(selector); if (node) node.textContent = text; }};

  function setControlDisabled(selector, disabled) {{
    const control = $(selector);
    if (control) control.disabled = disabled;
  }}

  function clearTranscript() {{
    transcriptEntries.clear();
    const list = $("#lk-transcript-list");
    if (!list) return;
    list.replaceChildren();
    const empty = document.createElement("div");
    empty.id = "lk-transcript-empty";
    empty.className = "transcript-empty";
    empty.textContent = "Your conversation will appear here.";
    list.appendChild(empty);
  }}

  function appendTranscript(text, isUser, isFinal = true, segmentId = "") {{
    const cleanText = String(text || "").trim();
    if (!cleanText) return;
    const list = $("#lk-transcript-list");
    if (!list) return;
    $("#lk-transcript-empty")?.remove();
    const role = isUser ? "user" : "agent";
    const key = segmentId || `${{role}}-${{Date.now()}}-${{Math.random()}}`;
    let entry = transcriptEntries.get(key);
    if (!entry) {{
      const wrapper = document.createElement("div");
      wrapper.className = `transcript-entry ${{role}}`;
      const speaker = document.createElement("div");
      speaker.className = "transcript-speaker";
      speaker.textContent = isUser ? "You" : "Agent";
      const bubble = document.createElement("div");
      bubble.className = "transcript-bubble";
      wrapper.append(speaker, bubble);
      list.appendChild(wrapper);
      entry = {{ wrapper, bubble }};
      transcriptEntries.set(key, entry);
    }}
    entry.bubble.textContent = cleanText;
    entry.wrapper.classList.toggle("interim", !isFinal);
    if (isFinal) transcriptEntries.delete(key);
    list.scrollTop = list.scrollHeight;
  }}

  function setSpeakerMuted(muted) {{
    speakerMuted = muted;
    $("#lk-audio-sink")?.querySelectorAll("audio, video").forEach((element) => {{ element.muted = muted; }});
    const button = $("#lk-speaker-button");
    if (button) button.textContent = muted ? "Unmute speaker" : "Mute speaker";
  }}

  function readGradioValue(selector) {{
    const root = $(selector);
    if (!root) return "";
    const field = root.matches("textarea, input") ? root : root.querySelector("textarea, input");
    return field ? field.value : (root.dataset.value || "");
  }}

  function setConsoleState(label, live = false) {{
    const badge = $("#lk-connection-status");
    if (badge) {{
      badge.classList.toggle("live", live);
      badge.innerHTML = `<span class="state-dot"></span>${{label}}`;
    }}
  }}

  function setListeningState(label, live = false) {{
    const badge = $("#lk-listening-status");
    if (badge) {{
      badge.classList.toggle("live", live);
      badge.innerHTML = `<span class="state-dot"></span>${{label}}`;
    }}
  }}

  function setConsoleCopy(title, subtitle) {{
    setText("#lk-console-copy", title);
    setText("#lk-console-subcopy", subtitle);
  }}

  function appendLog(message) {{
    const log = $("#lk-event-log");
    if (!log) return;
    const stamp = new Date().toLocaleTimeString([], {{ hour: "2-digit", minute: "2-digit" }});
    log.textContent = `${{stamp}}  ·  ${{message}}`;
  }}

  async function loadClient() {{
    if (window.LivekitClient || window.LiveKitClient) return window.LivekitClient || window.LiveKitClient;
    if (loadingClient) return loadingClient;
    loadingClient = new Promise((resolve, reject) => {{
      const script = document.createElement("script");
      script.src = CDN;
      script.async = true;
      script.onload = () => resolve(window.LivekitClient || window.LiveKitClient);
      script.onerror = () => reject(new Error("Could not load the LiveKit browser client."));
      document.head.appendChild(script);
    }});
    return loadingClient;
  }}

  function clearAudio() {{
    const sink = $("#lk-audio-sink");
    if (sink) sink.replaceChildren();
  }}

  function updateParticipantCount() {{
    const count = room ? room.remoteParticipants.size : 0;
    setText("#lk-session-meta", room ? `${{count + 1}} participant${{count === 0 ? "" : "s"}} in room` : "No active room");
  }}

  async function disconnect(reason = "Disconnected") {{
    const activeRoom = room;
    room = null;
    if (activeRoom) {{ try {{ activeRoom.disconnect(); }} catch (_) {{}} }}
    microphoneEnabled = false;
    speakerMuted = false;
    clearAudio();
    clearTranscript();
    const ended = reason === "Session ended";
    setConsoleState(ended ? "Ended" : "Ready", false);
    setListeningState("Muted", false);
    setConsoleCopy(
      ended ? "Session ended safely" : "Welcome — your session is ready",
      ended ? "Whenever you’re ready, start a new session on the right." : "Start a session on the right, then enable your microphone.",
    );
    updateParticipantCount();
    setText("#lk-room-id", "Room not connected");
    appendLog(reason);
    const mic = $("#lk-mic-button");
    if (mic) {{ mic.textContent = "Unmute microphone"; mic.classList.remove("primary"); }}
    const speaker = $("#lk-speaker-button");
    if (speaker) speaker.textContent = "Mute speaker";
    setControlDisabled("#lk-mic-button", true);
    setControlDisabled("#lk-speaker-button", true);
    setControlDisabled("#lk-disconnect-button", true);
    const orb = $("#lk-voice-orb");
    if (orb) orb.classList.remove("speaking");
  }}

  async function connect(url, token) {{
    try {{
      await disconnect("Opening LiveKit room");
      setConsoleState("Connecting", false);
      setConsoleCopy("Connecting to LiveKit…", "Preparing the voice channel and agent dispatch.");
      appendLog("Connecting to room");
      const client = await loadClient();
      if (!client) throw new Error("LiveKit browser client is unavailable.");
      room = new client.Room({{ adaptiveStream: true, dynacast: true }});
      room.on("trackSubscribed", (track) => {{
        if (track.kind === "audio") {{
          const element = track.attach();
          element.autoplay = true;
          element.muted = speakerMuted;
          $("#lk-audio-sink")?.appendChild(element);
          appendLog("Agent audio connected");
        }}
      }});
      room.on("trackUnsubscribed", (track) => track.detach());
      room.on("participantConnected", () => {{ updateParticipantCount(); appendLog("Agent joined the room"); }});
      room.on("participantDisconnected", () => {{ updateParticipantCount(); appendLog("Participant left the room"); }});
      room.on("activeSpeakersChanged", (speakers) => {{
        const speaking = speakers && speakers.length > 0;
        $("#lk-voice-orb")?.classList.toggle("speaking", speaking);
      }});
      room.on("disconnected", () => {{ disconnect("LiveKit room disconnected"); }});
      room.on("dataReceived", (payload, participant, kind, topic) => {{
        try {{
          const message = new TextDecoder().decode(payload);
          appendLog(`${{topic || "Data"}}: ${{message.slice(0, 90)}}`);
        }} catch (_) {{ appendLog("Received a LiveKit data message"); }}
      }});
      if (room.registerTextStreamHandler) {{
        room.registerTextStreamHandler("lk.transcription", async (reader, participantInfo) => {{
          try {{
            const message = await reader.readAll();
            const attributes = reader.info?.attributes || {{}};
            const isUser = participantInfo?.identity === room.localParticipant.identity;
            const isFinal = attributes["lk.transcription_final"] !== "false";
            const segmentId = `${{isUser ? "user" : "agent"}}-${{attributes["lk.segment_id"] || Date.now()}}`;
            appendTranscript(message, isUser, isFinal, segmentId);
            appendLog(`${{isUser ? "Speech recognized" : "Agent transcript"}}: ${{message.slice(0, 70)}}`);
          }} catch (error) {{ appendLog(error?.message || "Could not read the transcript"); }}
        }});
      }} else {{
        room.on("transcriptionReceived", (segments, participant) => {{
          (segments || []).forEach((segment) => {{
            const isUser = participant?.identity === room.localParticipant.identity;
            appendTranscript(segment.text, isUser, segment.final ?? true, `${{isUser ? "user" : "agent"}}-${{segment.id || segment.segmentId || Date.now()}}`);
          }});
        }});
      }}
      await room.connect(url, token);
      await room.localParticipant.setMicrophoneEnabled(false);
      microphoneEnabled = false;
      setControlDisabled("#lk-mic-button", false);
      setControlDisabled("#lk-speaker-button", false);
      setControlDisabled("#lk-disconnect-button", false);
      setSpeakerMuted(false);
      setConsoleState("Connected", true);
      setListeningState("Muted", false);
      setText("#lk-room-id", `Room: ${{readGradioValue("#lk-room-input")}}`);
      setConsoleCopy("Go ahead — just talk.", "Unmute your microphone and speak naturally to the voice agent.");
      updateParticipantCount();
      appendLog("Connected · microphone is off");
    }} catch (error) {{
      room = null;
      setConsoleState("Connection error", false);
      setConsoleCopy("Could not connect", error?.message || "Check the LiveKit URL and token.");
      appendLog(error?.message || "LiveKit connection failed");
    }}
  }}

  async function toggleMicrophone() {{
    if (!room) {{ appendLog("Connect a room first"); return; }}
    try {{
      await room.startAudio().catch(() => {{}});
      microphoneEnabled = !microphoneEnabled;
      await room.localParticipant.setMicrophoneEnabled(microphoneEnabled);
      const mic = $("#lk-mic-button");
      if (mic) {{
        mic.textContent = microphoneEnabled ? "Mute" : "Unmute microphone";
        mic.classList.toggle("primary", microphoneEnabled);
      }}
      setListeningState(microphoneEnabled ? "Listening" : "Muted", microphoneEnabled);
      setConsoleCopy(microphoneEnabled ? "Go ahead — just talk." : "You’re connected", microphoneEnabled ? "Your speech is being transcribed live below." : "Unmute your microphone whenever you’re ready.");
      appendLog(microphoneEnabled ? "Microphone enabled" : "Microphone muted");
    }} catch (error) {{ appendLog(error?.message || "Microphone permission was not granted"); }}
  }}

  function toggleSpeaker() {{
    if (!room) {{ appendLog("Connect a room first"); return; }}
    setSpeakerMuted(!speakerMuted);
    appendLog(speakerMuted ? "Speaker muted" : "Speaker unmuted");
  }}

  async function sendDataMessage() {{
    if (!room) {{ appendLog("Connect a room first"); return; }}
    const input = $("#lk-data-input");
    const text = input?.value.trim();
    if (!text) return;
    try {{
      if (!room.localParticipant.sendText) throw new Error("Text input is unavailable in this LiveKit client.");
      await room.localParticipant.sendText(text, {{ topic: "lk.chat" }});
      if (input) input.value = "";
      appendTranscript(text, true, true, `typed-${{Date.now()}}`);
      appendLog(`Message sent: ${{text.slice(0, 70)}}`);
    }} catch (error) {{ appendLog(error?.message || "Could not send data signal"); }}
  }}

  function wireConsole() {{
    const mic = $("#lk-mic-button");
    if (!mic || mic.dataset.wired === "1") return;
    mic.dataset.wired = "1";
    mic.addEventListener("click", toggleMicrophone);
    $("#lk-speaker-button")?.addEventListener("click", toggleSpeaker);
    $("#lk-disconnect-button")?.addEventListener("click", () => disconnect("Disconnected by operator"));
    $("#lk-send-button")?.addEventListener("click", sendDataMessage);
    $("#lk-clear-transcript")?.addEventListener("click", clearTranscript);
    $("#lk-data-input")?.addEventListener("keydown", (event) => {{ if (event.key === "Enter") sendDataMessage(); }});
  }}

  function syncFromGradio() {{
    const token = readGradioValue("#lk-token-state");
    const url = readGradioValue("#lk-url-state") || readGradioValue("#lk-url-input");
    if (token && token !== observedToken) {{
      observedToken = token;
      connect(url, token);
    }} else if (!token && observedToken) {{
      observedToken = "";
      disconnect("Session ended");
    }}
  }}

  const boot = () => {{
    wireConsole();
    window.setInterval(syncFromGradio, 450);
    const observer = new MutationObserver(() => wireConsole());
    observer.observe(document.body, {{ childList: true, subtree: true }});
  }};
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
}}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="LiveKit Voice Lab",
    ) as demo:
        gr.HTML(
            """
            <div class="app-shell hero">
              <div>
                <div class="eyebrow">LiveKit / browser voice workspace</div>
                <h1>Voice agent</h1>
                <p>A LiveKit voice agent you can talk to in the browser — deepgram/nova-3 for hearing, google/gemma-4-31b-it for thinking, inworld/inworld-tts-2 (Ashley) for speaking.</p>
              </div>
              <div class="hero-mark"><span class="mark">◉</span> Agent workspace</div>
            </div>
            """
        )

        # The token is deliberately kept out of the visible interface. The
        # browser-side LiveKit console reads this state after a Python callback
        # issues a room token.
        token_state = gr.Textbox(
            value="",
            show_label=False,
            container=False,
            elem_id="lk-token-state",
            elem_classes="lk-internal-state",
        )
        url_state = gr.Textbox(
            value=os.getenv("LIVEKIT_URL", ""),
            show_label=False,
            container=False,
            elem_id="lk-url-state",
            elem_classes="lk-internal-state",
        )

        with gr.Row(elem_classes="app-shell"):
            with gr.Column(scale=8, elem_classes="main-panel"):
                with gr.Group(elem_classes="surface session-surface"):
                    gr.HTML(
                        """
                        <div id="lk-console-root">
                          <div class="session-top">
                            <div>
                              <div class="section-kicker">Live session</div>
                              <div class="session-title">Your live conversation</div>
                            </div>
                            <div class="session-status-group">
                              <div id="lk-connection-status" class="session-state"><span class="state-dot"></span>Ready</div>
                              <div id="lk-listening-status" class="session-state"><span class="state-dot"></span>Muted</div>
                              <div id="lk-room-id" class="console-subcopy">Room not connected</div>
                            </div>
                          </div>
                          <div class="live-console">
                            <div class="console-visual">
                              <div>
                                <div id="lk-voice-orb" class="voice-orb">🎙</div>
                                <div id="lk-console-copy" class="console-copy">Go ahead — just talk.</div>
                                <div id="lk-console-subcopy" class="console-subcopy">Start a session, then unmute your microphone when you’re ready.</div>
                              </div>
                            </div>
                            <div class="console-actions">
                              <button id="lk-mic-button" class="primary call-control" type="button" disabled>Unmute microphone</button>
                              <button id="lk-speaker-button" class="call-control" type="button" disabled>Mute speaker</button>
                              <button id="lk-disconnect-button" class="danger call-control" type="button" disabled>End call</button>
                              <span id="lk-session-meta" class="console-subcopy">No active room</span>
                            </div>
                            <div class="data-row">
                              <input id="lk-data-input" type="text" placeholder="Type a message if you prefer…" aria-label="Type a message to the agent" />
                              <button id="lk-send-button" type="button">Send message</button>
                            </div>
                            <div class="transcript-panel">
                              <div class="transcript-header">
                                <div>
                                  <div class="transcript-title">Live transcript</div>
                                  <div class="transcript-caption">Speech-to-text and agent replies appear here.</div>
                                </div>
                                <button id="lk-clear-transcript" class="transcript-clear" type="button">Clear</button>
                              </div>
                              <div id="lk-transcript-list" class="transcript-list">
                                <div id="lk-transcript-empty" class="transcript-empty">Your conversation will appear here.</div>
                              </div>
                            </div>
                            <div id="lk-event-log" class="event-log">Waiting for you to start a session.</div>
                            <div id="lk-audio-sink" aria-hidden="true"></div>
                          </div>
                        </div>
                        """
                    )

            with gr.Column(scale=4, elem_classes="sidebar-panel right-stack"):
                with gr.Group(elem_classes="surface control-surface"):
                    gr.HTML(
                        """
                        <div class="control-title">Start a session</div>
                        <div class="control-description">We’ll start the agent, create a secure room connection, and keep your microphone muted until you choose to speak.</div>
                        """
                    )
                    room_input = gr.Textbox(
                        label="Room name",
                        value=DEFAULT_ROOM,
                        placeholder="e.g. product-demo",
                        elem_id="lk-room-input",
                    )
                    identity_input = gr.Textbox(
                        label="Your name",
                        value=DEFAULT_IDENTITY,
                        placeholder="e.g. Rizwan",
                    )
                    url_input = gr.Textbox(
                        label="LiveKit URL",
                        value=os.getenv("LIVEKIT_URL", ""),
                        placeholder="wss://your-project.livekit.cloud",
                        elem_id="lk-url-input",
                    )
                    with gr.Row():
                        start_session_button = gr.Button("Start session", variant="primary")
                        end_session_button = gr.Button("End session")
                    connection_state = gr.Markdown(f"**Session ready** · room `{DEFAULT_ROOM}`, joining as `{DEFAULT_IDENTITY}`.")
                    connection_hint = gr.Markdown("Allow microphone access when your browser asks. The agent will join automatically after you start the session.", elem_classes="hint")

                with gr.Group(elem_classes="surface flow-surface"):
                    gr.HTML(
                        """
                        <div class="flow-title">How it works</div>
                        <div class="flow-step"><span>1</span><div>Start a session to bring the voice agent online.</div></div>
                        <div class="flow-step"><span>2</span><div>Connect to the room with your browser.</div></div>
                        <div class="flow-step"><span>3</span><div>Enable your microphone and start talking.</div></div>
                        """
                    )

                with gr.Accordion("Advanced worker controls", open=False, elem_classes="advanced-panel"):
                    gr.HTML(
                        """
                        <div class="control-title">Agent worker</div>
                        <div class="control-description">Most sessions only need the buttons above. Use these controls when you want to manage the <code>agent.py</code> worker separately.</div>
                        """
                    )
                    worker_status = gr.Markdown(worker_status_markdown())
                    with gr.Row(elem_classes="worker-buttons"):
                        start_button = gr.Button("Start worker only", variant="primary")
                        restart_button = gr.Button("Restart worker")
                        stop_button = gr.Button("Stop worker")
                    activity = gr.Markdown("Ready.", elem_classes="hint")
                    gr.HTML(
                        f"<div class='config-note'>Registered agent <code>{AGENT_NAME}</code> · room-level dispatch is included in each generated token.</div>"
                    )

        refresh_timer = gr.Timer(3)

        start_session_button.click(
            fn=start_worker,
            outputs=[worker_status, activity],
        ).then(
            fn=connect_session,
            inputs=[room_input, identity_input, url_input],
            outputs=[token_state, url_state, connection_state, connection_hint],
        )
        end_session_button.click(
            fn=disconnect_session,
            outputs=[token_state, url_state, connection_state, connection_hint],
        ).then(
            fn=stop_worker,
            outputs=[worker_status, activity],
        )
        start_button.click(fn=start_worker, outputs=[worker_status, activity])
        restart_button.click(fn=restart_worker, outputs=[worker_status, activity])
        stop_button.click(fn=stop_worker, outputs=[worker_status, activity])
        refresh_timer.tick(fn=worker_status_markdown, outputs=worker_status)

    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="sky",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        ),
        css=APP_CSS,
        js=LIVEKIT_CLIENT_JS,
        show_error=True,
    )
