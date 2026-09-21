"""LiveKit voice worker used by the browser voice workspace in app.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, inference, room_io
from livekit.plugins import ai_coustics


load_dotenv(Path(__file__).with_name(".env"))

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "my-agent").strip() or "my-agent"
STT_MODEL = os.getenv("LIVEKIT_STT_MODEL", "deepgram/nova-3").strip()
STT_LANGUAGE = os.getenv("LIVEKIT_STT_LANGUAGE", "multi").strip() or "multi"
LLM_MODEL = os.getenv("LIVEKIT_LLM_MODEL", "google/gemma-4-31b-it").strip()
TTS_MODEL = os.getenv("LIVEKIT_TTS_MODEL", "inworld/inworld-tts-2").strip()
TTS_VOICE = os.getenv("LIVEKIT_TTS_VOICE", "Ashley").strip() or "Ashley"


class Assistant(Agent):
    """Concise, friendly assistant persona used by each voice session."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. Be warm, curious, and concise. "
                "Answer directly in natural spoken language. Avoid markdown, emojis, "
                "asterisks, and decorative punctuation because your responses are spoken aloud. "
                "Ask a brief follow-up question when it would help."
            ),
        )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: agents.JobContext) -> None:
    """Start the configured speech-to-speech pipeline for one room."""

    session = AgentSession(
        stt=inference.STT(model=STT_MODEL, language=STT_LANGUAGE),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(model=TTS_MODEL, voice=TTS_VOICE),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S,
                ),
            ),
            text_output=room_io.TextOutputOptions(
                sync_transcription=True,
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and offer your assistance in one short sentence.",
    )


if __name__ == "__main__":
    # LiveKit Agents 1.8+ expects an explicit CLI command. Keeping the
    # default as start makes python agent.py convenient locally.
    if len(sys.argv) == 1:
        sys.argv.append("start")
    agents.cli.run_app(server)
