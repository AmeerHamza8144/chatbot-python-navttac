# Chatbot Workspace

A local AI chatbot workspace with two interfaces:

1. `main.py` — a Streamlit text, coding, document, image, and GitHub assistant powered by Ollama.
2. `app.py` — a professional Gradio control panel for starting and using the LiveKit voice agent defined in `agent.py`.

The two applications are independent. Use `main.py` for local text/code work and `app.py` for browser-based voice conversations.

## Project architecture

```text
Browser
  ├── main.py  ── Streamlit ── Ollama local models
  └── app.py   ── Gradio ───── LiveKit room
                                  │
                                  └── agent.py worker
                                        ├── Speech-to-text
                                        ├── LLM response
                                        ├── Text-to-speech
                                        └── Noise cancellation
```

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Main Streamlit AI workspace. Uses Ollama and local conversation storage. |
| `app.py` | Gradio LiveKit dashboard. Starts the worker, creates room tokens, and provides browser voice controls. |
| `agent.py` | LiveKit `AgentServer` worker. Defines the voice assistant and its STT, LLM, TTS, and audio settings. |
| `pyproject.toml` | Project metadata and core dependencies. |
| `uv.lock` | Locked dependency versions for `uv`. |
| `.env` | Local secrets and LiveKit configuration. Do not commit this file. |
| `conversations.json` | Conversation history used by `main.py`. |
| `conversation.json` | Existing legacy or auxiliary conversation data. The current `main.py` uses `conversations.json`. |
| `venv/` | Windows virtual environment already present in this workspace. |
| `.venv/` | An additional virtual environment directory. Use only one environment consistently. |

## Requirements

- Python 3.13 or newer. The project declares `requires-python = ">=3.13"`.
- Windows, macOS, or Linux.
- Internet access for LiveKit Cloud and the browser LiveKit client.
- Ollama installed and running for `main.py`.
- LiveKit project credentials for `app.py` and `agent.py`.
- A modern browser with microphone permission for the voice interface.

## 1. Open the project

PowerShell:

```powershell
cd "C:\Users\Ameer Hamza\chatbot"
```

Command Prompt:

```cmd
cd /d "C:\Users\Ameer Hamza\chatbot"
```

## 2. Activate the virtual environment

The commands below use the existing `venv` directory.

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current terminal, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
venv\Scripts\activate.bat
```

### macOS or Linux

If you create a Unix virtual environment named `.venv`:

```bash
source .venv/bin/activate
```

### Confirm activation

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

The executable path should point inside `venv` or `.venv`.

To leave the environment later:

```powershell
deactivate
```

## 3. Install dependencies

Upgrade packaging tools first:

```powershell
python -m pip install --upgrade pip
```

Install the dependencies declared in the project:

```powershell
python -m pip install -e .
```

This covers the Gradio and LiveKit application dependencies from `pyproject.toml`.

`main.py` also imports Streamlit, Ollama, document-processing, image-processing, and GitHub packages. Install these if they are not already present:

```powershell
python -m pip install streamlit ollama pillow pypdf pytesseract python-docx PyGithub
```

### Using `uv`

If `uv` is installed, the lock file can be used instead:

```powershell
uv sync
uv run python app.py
```

For the Streamlit application:

```powershell
uv run streamlit run main.py
```

Use either the `pip` workflow or the `uv` workflow for an environment, rather than mixing environments accidentally.

## 4. Configure environment variables

Create or edit `.env` in the project root. It must contain:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

Do not put real secrets in this README, source code, screenshots, or Git commits.

### What each variable does

| Variable | Used for |
| --- | --- |
| `LIVEKIT_URL` | LiveKit WebSocket endpoint used by the browser and agent worker. |
| `LIVEKIT_API_KEY` | Server-side token generation and LiveKit worker authentication. |
| `LIVEKIT_API_SECRET` | Server-side signing of access tokens. Keep this private. |

The dashboard loads `.env` automatically through `python-dotenv`.

## 5. Start the text and coding assistant

Make sure Ollama is installed and running.

If Ollama is not already running, start it in a separate terminal:

```powershell
ollama serve
```

Pull at least the default model:

```powershell
ollama pull qwen2.5-coder:7b
```

Available model names in `main.py` are:

- `qwen2.5-coder:7b` — default coding model.
- `deepseek-r1:latest` — reasoning model.
- `gemma3:1b` — smaller general model.
- `qwen3.5:2b` — compact model.

Start Streamlit:

```powershell
python -m streamlit run main.py
```

Open the URL shown in the terminal, normally:

```text
http://localhost:8501
```

### What `main.py` does

- Provides a multi-chat Streamlit interface.
- Saves chats to `conversations.json`.
- Lets you create, switch, and delete conversations.
- Checks whether the selected Ollama model is installed.
- Sends prompts to Ollama using `ollama.chat`.
- Supports PDF, DOCX, TXT, Markdown, CSV, JSON, image, and ZIP uploads.
- Extracts text from documents and sends it as model context.
- Uses OCR through `pytesseract` for image text extraction.
- Sends supported images to Ollama for image-aware responses.
- Limits uploaded file context to approximately 20,000 characters.
- Allows individual uploads up to 500 MB through the Streamlit interface.
- Shows model thinking when the selected model returns it.
- Provides optional GitHub Agent Mode.

### GitHub Agent Mode

In the Streamlit sidebar:

1. Open **GitHub Integration**.
2. Enable **GitHub Agent Mode**.
3. Enter a GitHub personal access token.
4. Enter a repository in the form `owner/repository`.

The current implementation reads the repository structure and gives that context to the model. It does not currently commit or push changes to GitHub.

### Image OCR requirement

The Python package `pytesseract` is only a wrapper. For OCR to work, the Tesseract executable must also be installed on the operating system and available on `PATH`.

## 6. Start the LiveKit voice application

Run:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

### Friendly session flow

1. Enter or confirm the LiveKit URL, room name, and participant identity.
2. Click **Start session**.
3. The dashboard starts the `agent.py` worker and creates a secure LiveKit room token.
4. The browser connects to the LiveKit room.
5. Click **Enable microphone** when you are ready to speak.
6. Talk to the agent and listen to its response in the browser.
7. Click **End session** when finished.

The microphone is intentionally muted when the room first connects. This prevents audio from being published before the user explicitly enables it.

### What `app.py` does

- Runs the Gradio control panel on port `7860`.
- Reads LiveKit credentials from `.env`.
- Starts the LiveKit worker defined by `agent.py` in a managed subprocess.
- Creates an access token for the selected room and participant.
- Includes an explicit `my-agent` room dispatch in the token.
- Loads the LiveKit browser client from the public CDN.
- Connects the browser to LiveKit.
- Publishes microphone audio after permission is granted.
- Plays subscribed agent audio in the browser.
- Shows connection, participant, and activity information.
- Provides optional LiveKit data signals.
- Stops the managed worker when **End session** is clicked.
- Provides advanced worker controls for starting, restarting, or stopping the worker separately.

The browser-side LiveKit client requires internet access to load the CDN script. The LiveKit URL and API credentials must also be valid.

### Optional data signals

The **Send signal** field publishes a LiveKit data message with the topic `gradio.command`. The current `agent.py` is configured primarily for voice input and does not define a text/data-message handler, so voice is the supported way to talk with the agent.

## 7. Understanding `agent.py`

`agent.py` defines the actual LiveKit voice worker.

### Agent identity

The registered agent name is:

```text
my-agent
```

### Voice pipeline

The worker configures:

- Speech-to-text: `deepgram/nova-3`, multilingual mode.
- Language model: `google/gemma-4-31b-it`.
- Text-to-speech: `inworld/inworld-tts-2` with the `Ashley` voice.
- Turn detection: LiveKit turn detector.
- Noise cancellation: ai-coustics `QUAIL_VF_S` enhancement.

When a session starts, the agent greets the user automatically and then waits for voice input.

### Running `agent.py` directly

You can run the worker directly with:

```powershell
python agent.py
```

When run directly, it also starts the legacy Gradio dashboard defined inside `agent.py` on port `7860`. For normal use, prefer `python app.py`, because `app.py` provides the newer interface and manages the worker for you.

Do not run `agent.py` directly and `app.py` at the same time unless you intentionally change the ports and worker setup. Both may try to use port `7860`, and duplicate workers can receive the same room dispatch.

## Which command should I use?

| Goal | Command |
| --- | --- |
| Text chat, coding, documents, images, or Ollama | `python -m streamlit run main.py` |
| Friendly LiveKit voice interface | `python app.py` |
| Direct legacy LiveKit worker/dashboard | `python agent.py` |

## Stopping the applications

In the terminal running the application, press:

```text
Ctrl+C
```

For the LiveKit dashboard, clicking **End session** also disconnects the browser and stops the worker managed by `app.py`.

Stopping `main.py` does not stop Ollama. If you started Ollama manually with `ollama serve`, stop that terminal separately when you are finished.

## Troubleshooting

### `ModuleNotFoundError`

Activate the correct environment and install the missing package:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install <package-name>
```

For the common `main.py` packages:

```powershell
python -m pip install streamlit ollama pillow pypdf pytesseract python-docx PyGithub
```

### Ollama is offline

Start Ollama:

```powershell
ollama serve
```

Confirm a model is installed:

```powershell
ollama list
```

Pull one if needed:

```powershell
ollama pull qwen2.5-coder:7b
```

### LiveKit credentials are missing

Confirm `.env` exists in the same directory as `app.py` and contains all three variables:

```env
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Restart `app.py` after changing `.env`.

### Port `7860` is already in use

Stop another Gradio or LiveKit dashboard process before starting `app.py`. The direct `agent.py` mode also uses this port.

Streamlit normally uses port `8501`. To choose another Streamlit port:

```powershell
python -m streamlit run main.py --server.port 8502
```

### The microphone does not work

- Allow microphone access when the browser asks.
- Use `http://127.0.0.1:7860` or `http://localhost:7860`.
- Confirm another application is not exclusively using the microphone.
- Click **Enable microphone** after the LiveKit connection says it is ready.
- Check that the browser can reach the LiveKit URL.

### The worker is offline

Open **Advanced worker controls** and check the status card. Verify the LiveKit credentials, then restart the worker. If the worker exits immediately, run `python agent.py` in a separate terminal to view its startup error.

### The agent connects but does not respond

Check all of the following:

- The worker status is online.
- The browser microphone is enabled.
- The LiveKit project is reachable.
- The LiveKit inference providers and models are available to the project.
- The browser is not blocking remote audio playback.

## Security notes

- Never commit `.env` or share `LIVEKIT_API_SECRET`.
- Keep API credentials on the server side. `app.py` sends only a room access token to the browser.
- Use a restricted GitHub token for GitHub Agent Mode.
- Review uploaded files before sending sensitive content to any model or external service.
- The default Gradio and Streamlit servers are intended for local development. Do not expose them publicly without authentication, HTTPS, and a deployment plan.

## Development notes

- The project uses Python type hints and Python 3.13 syntax.
- The LiveKit worker and Gradio UI are separate processes when started through `app.py`.
- `app.py` owns the worker process it starts and stops only that process.
- `main.py` persists chat history after user and assistant messages.
- Changes to `.env` require restarting the relevant application.
- Changes to `agent.py` require restarting the LiveKit worker.
