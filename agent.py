import os
from pathlib import Path
import sys
import threading
import time
from dotenv import load_dotenv

# LiveKit imports (and plugin registrations) happen on the MAIN thread here
from livekit import agents, api
from livekit.agents import AgentServer, AgentSession, Agent, inference, room_io, TurnHandlingOptions
from livekit.plugins import ai_coustics

import gradio as gr

load_dotenv(Path(__file__).with_name(".env"))

# ==============================================================================
# LIVEKIT AGENT WORKER DEFINITION
# ==============================================================================

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

server = AgentServer()

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="inworld/inworld-tts-2",
            voice="Ashley",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_S),
            ),
            text_output=room_io.TextOutputOptions(
                sync_transcription=False,
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


# ==============================================================================
# GRADIO DASHBOARD & HELPER FUNCTIONS
# ==============================================================================

def generate_livekit_token(room_name: str, participant_identity: str) -> str:
    """Generates an Access Token for room connection testing."""
    api_key = os.getenv("LIVEKIT_API_KEY", "")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "")

    if not api_key or not api_secret:
        return "Error: LIVEKIT_API_KEY or LIVEKIT_API_SECRET missing from environment variables."

    if not room_name or not participant_identity:
        return "Error: Room Name and Participant Identity are required."

    try:
        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(participant_identity)
            .with_name(participant_identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
        )
        return token.to_jwt()
    except Exception as exc:
        return f"Error generating token: {str(exc)}"


def get_worker_status() -> str:
    """Checks environment variables and worker status."""
    url = os.getenv("LIVEKIT_URL", "Not Configured")
    key = "Configured" if os.getenv("LIVEKIT_API_KEY") else "Missing"
    secret = "Configured" if os.getenv("LIVEKIT_API_SECRET") else "Missing"
    
    return f"""### Worker Status: Active 🟢
- **Server URL:** `{url}`
- **API Key:** `{key}`
- **API Secret:** `{secret}`
- **Registered Agent:** `my-agent`
- **Uptime:** Active on main thread
"""


def build_dashboard() -> gr.Blocks:
    """Constructs the Gradio web control panel."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    )

    with gr.Blocks(theme=theme, title="LiveKit Agent Control Panel") as demo:
        gr.Markdown(
            """
            # 🎙️ LiveKit Voice Agent Dashboard
            Manage connections, generate tokens, and inspect background worker configurations.
            """
        )

        with gr.Tabs():
            # Tab 1: Token Generator & Playground
            with gr.TabItem("🔑 Token Generator & Playground"):
                gr.Markdown("### Room Token Generator")
                with gr.Row():
                    room_input = gr.Textbox(
                        label="Room Name",
                        value="test-room",
                        placeholder="Enter room name..."
                    )
                    identity_input = gr.Textbox(
                        label="Participant Identity",
                        value="user-1",
                        placeholder="Enter user identity..."
                    )

                gen_btn = gr.Button("Generate Token", variant="primary")
                token_output = gr.Textbox(
                    label="Generated JWT Access Token",
                    interactive=False,
                    lines=3
                )

                gen_btn.click(
                    fn=generate_livekit_token,
                    inputs=[room_input, identity_input],
                    outputs=token_output
                )

                gr.Markdown("---")
                gr.Markdown("### 🧪 Quick Web Playground Link")
                playground_url_output = gr.Markdown("Generate a token above to get the LiveKit Sandbox testing link.")

                def generate_sandbox_link(room_name: str, identity: str) -> str:
                    token = generate_livekit_token(room_name, identity)
                    if token.startswith("Error"):
                        return f"**Error:** {token}"
                    livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
                    return f"👉 **[Click to Open LiveKit Web Sandbox Playground](https://agents-playground.livekit.io/?livekitUrl={livekit_url}&token={token})**"

                gen_btn.click(
                    fn=generate_sandbox_link,
                    inputs=[room_input, identity_input],
                    outputs=playground_url_output
                )

            # Tab 2: Worker Status
            with gr.TabItem("📊 Worker Status"):
                status_display = gr.Markdown(get_worker_status())
                refresh_btn = gr.Button("Refresh Status")
                refresh_btn.click(fn=get_worker_status, outputs=status_display)

    return demo


def run_gradio_background():
    """Launches Gradio non-blocking server in a background thread."""
    dashboard = build_dashboard()
    # launch(prevent_thread_lock=True) keeps Gradio running asynchronously in background
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,
    )


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    # LiveKit Agents 1.8+ requires an explicit CLI command. Default to the
    # production worker when this file is run without arguments.
    if len(sys.argv) == 1:
        sys.argv.append("start")

    # 1. Start Gradio in background thread
    gradio_thread = threading.Thread(target=run_gradio_background, daemon=True)
    gradio_thread.start()
    print("🚀 Gradio Dashboard launching on http://localhost:7860")

    # 2. Run LiveKit CLI app directly on the MAIN THREAD
    agents.cli.run_app(server)
