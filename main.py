import base64
import io
import json
import os
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

# Streamlit & AI dependencies
import ollama
from PIL import Image
from pypdf import PdfReader
import pytesseract
import streamlit as st

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from github import Github
except ImportError:
    Github = None

HISTORY_FILE = Path(__file__).with_name("conversations.json")
ALLOWED_MODELS = [
    "qwen2.5-coder:7b",
    "deepseek-r1:latest",
    "gemma3:1b",
    "qwen3.5:2b",
]
DEFAULT_MODEL = ALLOWED_MODELS[0]
MAX_CONTEXT_CHARACTERS = 20000
MAX_FILE_SIZE_MB = 500

SUPPORTED_UPLOAD_TYPES = [
    "pdf", "txt", "md", "csv", "json", "docx", "png", "jpg", "jpeg", "webp", "zip"
]

SUGGESTIONS = [
    {"title": "Write & Code", "desc": "Refactor C# / Python or build a web app", "icon": "💻"},
    {"title": "GitHub Agent", "desc": "Inspect repo, write code, and commit via API", "icon": "🐙"},
    {"title": "Analyze Data", "desc": "Summarize complex PDFs, docs or CSVs", "icon": "📊"},
    {"title": "Vision & Images", "desc": "Extract information and insights from images", "icon": "🖼️"},
]

SYSTEM_PROMPT = """You are a helpful, direct, and intelligent AI assistant.
- Provide well-structured, clear, and comprehensive answers using Markdown.
- If context or file attachments are provided, integrate them accurately into your response.
- Use clear visual headings, bullet points, and code blocks where applicable."""

GITHUB_AGENT_SYSTEM_PROMPT = """You are an autonomous GitHub coding agent. 
Your task is to inspect remote repositories, analyze code structures, and safely modify or create files.
When asked to perform a coding task across a repository:
1. List repository files to understand the project architecture.
2. Read required files to analyze dependencies and code patterns.
3. Propose and write complete, production-ready code blocks or execute direct commits if requested."""


# ==============================================================================
# DOCUMENT EXTRACTION HELPERS (PURE PYTHON)
# ==============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    page_texts = []
    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        if extracted.strip():
            page_texts.append(f"--- Page {index} ---\n{extracted.strip()}")
    return "\n\n".join(page_texts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    if Document is None:
        return "[Error: python-docx library is not installed]"
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)
    return "\n".join(paragraphs)


def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    except Exception as exc:
        return f"[OCR Error: {exc}]"


def extract_text_from_plain(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


# ==============================================================================
# STREAMLIT UI & LOGIC
# ==============================================================================

def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_timestamp(timestamp: str | None) -> str:
    if not timestamp:
        return ""
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return ""


def parse_thinking_process(raw_response: str) -> tuple[str, str]:
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, raw_response, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        cleaned_answer = re.sub(think_pattern, "", raw_response, flags=re.DOTALL).strip()
        return thinking, cleaned_answer
    return "", raw_response.strip()


def brainstorm_chat_title(prompt: str) -> str:
    clean_prompt = " ".join(prompt.strip().split())
    if len(clean_prompt) <= 30:
        return clean_prompt.capitalize()
    truncated = clean_prompt[:30].rsplit(" ", 1)[0]
    return f"{truncated.capitalize()}..."


# --------------------------------------------------------------------------
# Multi-Session & Live Persistence
# --------------------------------------------------------------------------

def load_all_conversations() -> dict:
    if not HISTORY_FILE.exists():
        return {}
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        st.warning("Could not load conversation history.")
    return {}


def save_all_conversations():
    try:
        payload = json.dumps(st.session_state.all_chats, ensure_ascii=False, indent=2)
        temp_file = HISTORY_FILE.with_suffix(".tmp")
        temp_file.write_text(payload, encoding="utf-8")
        temp_file.replace(HISTORY_FILE)
    except Exception as exc:
        st.error(f"Could not save conversation: {exc}")


def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.active_chat_id = chat_id
    st.session_state.all_chats[chat_id] = {
        "title": "New Chat",
        "created_at": current_timestamp(),
        "messages": [],
        "file_context": "",
        "file_images": [],
        "file_summary": "",
        "file_signature": []
    }
    save_all_conversations()


def switch_chat(chat_id: str):
    st.session_state.active_chat_id = chat_id


def delete_chat(chat_id: str):
    if chat_id in st.session_state.all_chats:
        del st.session_state.all_chats[chat_id]
        save_all_conversations()

    remaining_chats = list(st.session_state.all_chats.keys())
    if remaining_chats:
        st.session_state.active_chat_id = remaining_chats[0]
    else:
        create_new_chat()


# --------------------------------------------------------------------------
# GitHub API Integration Tools
# --------------------------------------------------------------------------

def get_github_repo(token: str, repo_name: str):
    if not Github or not token or not repo_name:
        return None
    try:
        gh = Github(token)
        return gh.get_repo(repo_name)
    except Exception:
        return None


def list_github_repo_structure(token: str, repo_name: str) -> str:
    repo = get_github_repo(token, repo_name)
    if not repo:
        return "GitHub repository connection failed or unconfigured."
    try:
        contents = repo.get_contents("")
        file_paths = []
        while contents:
            file_obj = contents.pop(0)
            if file_obj.type == "dir":
                contents.extend(repo.get_contents(file_obj.path))
            else:
                file_paths.append(file_obj.path)
        return "\n".join(file_paths[:100]) if file_paths else "Repository is empty."
    except Exception as exc:
        return f"Error listing repository structure: {exc}"


# --------------------------------------------------------------------------
# File Processing & Context Building
# --------------------------------------------------------------------------

def image_to_base64(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            rgb_img = img.convert("RGB")
            buffer = BytesIO()
            rgb_img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        st.warning(f"Image processing error: {exc}")
        return ""


def process_zip_bytes(zip_bytes: bytes):
    context_parts = []
    image_data = []

    try:
        with ZipFile(BytesIO(zip_bytes), "r") as zip_ref:
            for entry in sorted(zip_ref.infolist(), key=lambda item: item.filename):
                if entry.is_dir() or entry.filename.startswith("."):
                    continue

                filename = entry.filename
                lower_name = filename.lower()
                file_bytes = zip_ref.read(entry)

                if lower_name.endswith(".pdf"):
                    context_parts.append(f"[{filename}]\n{extract_text_from_pdf(file_bytes)}\n\n")
                elif lower_name.endswith((".txt", ".md", ".csv", ".json")):
                    text_content = extract_text_from_plain(file_bytes)
                    context_parts.append(f"[{filename}]\n{text_content}\n\n")
                elif lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    encoded = image_to_base64(file_bytes)
                    if encoded:
                        image_data.append(encoded)
    except (BadZipFile, OSError) as exc:
        st.warning(f"ZIP read error: {exc}")

    combined = "".join(context_parts)[:MAX_CONTEXT_CHARACTERS]
    return combined, image_data


def process_uploaded_file(file_obj):
    if file_obj is None:
        return "", []

    try:
        file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj
        filename = getattr(file_obj, "name", "uploaded_file")
        lower_name = filename.lower()

        if lower_name.endswith(".pdf"):
            return extract_text_from_pdf(file_bytes), []

        if lower_name.endswith(".docx"):
            return f"[{filename}]\n{extract_text_from_docx(file_bytes)}\n\n", []

        if lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            encoded = image_to_base64(file_bytes)
            return "", ([encoded] if encoded else [])

        if lower_name.endswith(".zip"):
            return process_zip_bytes(file_bytes)

        if lower_name.endswith((".txt", ".md", ".csv", ".json")):
            text_content = extract_text_from_plain(file_bytes)
            return f"[{filename}]\n{text_content}\n\n", []

    except Exception as exc:
        st.error(f"Error reading {getattr(file_obj, 'name', 'file')}: {exc}")

    return "", []


def process_uploaded_files(uploaded_files):
    context_parts = []
    image_data = []

    for uploaded_file in uploaded_files or []:
        context, images = process_uploaded_file(uploaded_file)
        filename = getattr(uploaded_file, "name", "attached file")
        if context:
            context_parts.append(context)
        elif images:
            context_parts.append(f"[{filename}]\nAn image is attached.")
        else:
            context_parts.append(f"[{filename}]\nCould not extract text.")
        image_data.extend(images)

    return "\n\n".join(context_parts)[:MAX_CONTEXT_CHARACTERS], image_data


def file_signature(uploaded_files) -> list:
    return [
        [getattr(f, "name", "file"), getattr(f, "size", 0)]
        for f in uploaded_files or []
    ]


def update_active_chat_file_context(uploaded_files) -> None:
    active_chat = st.session_state.all_chats[st.session_state.active_chat_id]
    signature = file_signature(uploaded_files)

    if signature == active_chat.get("file_signature", []):
        return

    if not uploaded_files:
        active_chat["file_context"] = ""
        active_chat["file_images"] = []
        active_chat["file_summary"] = ""
        active_chat["file_signature"] = []
        save_all_conversations()
        return

    context, images = process_uploaded_files(uploaded_files)
    names = [name for name, _ in signature]

    active_chat["file_context"] = context
    active_chat["file_images"] = images
    active_chat["file_summary"] = f"Attached {len(names)} item(s): " + ", ".join(names)
    active_chat["file_signature"] = signature
    save_all_conversations()


def clear_active_chat_file_context() -> None:
    active_chat = st.session_state.all_chats[st.session_state.active_chat_id]
    active_chat["file_context"] = ""
    active_chat["file_images"] = []
    active_chat["file_summary"] = ""
    active_chat["file_signature"] = []
    save_all_conversations()


# --------------------------------------------------------------------------
# Ollama Model Backend
# --------------------------------------------------------------------------

def get_model_info(model: str) -> bool:
    if model not in ALLOWED_MODELS:
        st.sidebar.warning("Selected model is not in allowed list.")
        return False

    try:
        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else []
        installed_models = {
            item.get("name") or item.get("model")
            for item in models
            if isinstance(item, dict) and (item.get("name") or item.get("model"))
        }

        if model in installed_models:
            st.sidebar.caption(f"🟢 **{model}** ready")
            return True

        st.sidebar.warning(f"Model `{model}` not installed. Run `ollama pull {model}`.")
        return False
    except Exception as exc:
        st.sidebar.error(f"Ollama offline: {exc}")
        return False


def query_model(prompt: str, model_name: str, system_prompt: str, files_context: str = "", image_data=None, github_context: str = ""):
    try:
        combined_prompt = prompt.strip()
        if github_context.strip():
            combined_prompt = f"{combined_prompt}\n\n--- GITHUB REPOSITORY CONTEXT ---\n{github_context}"
        if files_context.strip():
            combined_prompt = f"{combined_prompt}\n\n--- ATTACHED FILE CONTEXT ---\n{files_context}"

        user_message = {"role": "user", "content": combined_prompt}
        if image_data:
            user_message["images"] = image_data

        messages = [{"role": "system", "content": system_prompt}, user_message]

        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0.7, "num_ctx": 8192},
        )

        thinking = ""
        if isinstance(response, dict):
            message = response.get("message", {})
            thinking = message.get("thinking", "")
            answer = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        else:
            message = getattr(response, "message", None)
            thinking = getattr(message, "thinking", "") if message else ""
            answer = getattr(message, "content", "") if message is not None else ""

        raw_answer = str(answer or "").strip()
        parsed_thinking, cleaned_answer = parse_thinking_process(raw_answer)

        final_thinking = thinking or parsed_thinking
        return final_thinking, cleaned_answer if cleaned_answer else "[The model returned an empty response.]"
    except Exception as exc:
        st.error(f"Execution error: {exc}")
        return "", "[Error generating response. Ensure local Ollama instance is active.]"


# --------------------------------------------------------------------------
# UI Setup & CSS
# --------------------------------------------------------------------------

def initialize_session():
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = load_all_conversations()
    if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.all_chats:
        if st.session_state.all_chats:
            st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0]
        else:
            create_new_chat()

    active_chat = st.session_state.all_chats[st.session_state.active_chat_id]
    if "file_context" not in active_chat:
        active_chat["file_context"] = ""
    if "file_images" not in active_chat:
        active_chat["file_images"] = []
    if "file_summary" not in active_chat:
        active_chat["file_summary"] = ""
    if "file_signature" not in active_chat:
        active_chat["file_signature"] = []

    if "github_mode" not in st.session_state:
        st.session_state.github_mode = False


def submit_message(prompt: str, model_name: str):
    display_prompt = (prompt or "").strip() or "Analyze the attached content."
    active_chat = st.session_state.all_chats[st.session_state.active_chat_id]

    files_context = active_chat.get("file_context", "")
    image_data = active_chat.get("file_images", [])

    if not active_chat["messages"] or active_chat["title"] in {"New Chat", "New Conversation"}:
        active_chat["title"] = brainstorm_chat_title(display_prompt)

    active_chat["messages"].append({
        "role": "user",
        "content": display_prompt,
        "timestamp": current_timestamp(),
    })
    save_all_conversations()

    github_context = ""
    active_sys_prompt = SYSTEM_PROMPT
    if st.session_state.get("github_mode", False):
        active_sys_prompt = GITHUB_AGENT_SYSTEM_PROMPT
        token = st.session_state.get("gh_token", "")
        repo = st.session_state.get("gh_repo", "")
        if token and repo:
            file_structure = list_github_repo_structure(token, repo)
            github_context = f"Repository: {repo}\nFiles:\n{file_structure}"

    with st.spinner("Model is thinking and processing..."):
        thinking, answer = query_model(
            display_prompt,
            model_name,
            active_sys_prompt,
            files_context,
            image_data,
            github_context,
        )

    active_chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
        "timestamp": current_timestamp(),
    })
    save_all_conversations()


def render_copy_button(text_content: str, index: int):
    safe_text = json.dumps(text_content)
    st.components.v1.html(
        f"""
        <button id="cp-btn-{index}" onclick='copyFn_{index}()' style="
            background: #f0f4f9;
            border: 1px solid #dce3ed;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 500;
            cursor: pointer;
            color: #444746;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        ">
            📋 Copy
        </button>
        <script>
        function copyFn_{index}() {{
            const text = {safe_text};
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.getElementById("cp-btn-{index}");
                btn.innerHTML = "✓ Copied";
                setTimeout(() => {{ btn.innerHTML = "📋 Copy"; }}, 2000);
            }});
        }}
        </script>
        """,
        height=32,
    )


def inject_gemini_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8f9fa !important;
            color: #1f1f1f !important;
            font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: none !important;
            padding-right: 0px !important;
        }
        section[data-testid="stSidebar"] * {
            color: #1f1f1f !important;
        }
        .gemini-title {
            background: linear-gradient(135deg, #1a73e8 0%, #a142f4 50%, #d93025 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            margin-bottom: 0px;
        }
        .gemini-subtitle {
            color: #5f6368;
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        .main .block-container {
            padding-bottom: 120px !important;
            max-width: 860px !important;
        }
        .suggestion-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 16px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
            height: 100%;
        }
        .suggestion-card:hover {
            border-color: #a855f7;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(168, 85, 247, 0.08);
        }
        div[data-testid="stChatInput"] {
            border-radius: 28px !important;
            border: 1px solid #dce3ed !important;
            background-color: #ffffff !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.05) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main Application Flow
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Ameer Hamza Workspace", layout="wide", page_icon="✨")
    inject_gemini_styles()
    initialize_session()

    with st.sidebar:
        st.markdown("<h2 style='font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem; color: #1f1f1f;'>Hamza Workspace</h2>", unsafe_allow_html=True)
        
        if st.button("➕ New Chat", key="new_chat_top", use_container_width=True):
            create_new_chat()
            st.rerun()

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        selected_model = st.selectbox("Active Model", ALLOWED_MODELS, index=0)
        get_model_info(selected_model)

        with st.expander("🐙 GitHub Integration", expanded=False):
            st.session_state.github_mode = st.checkbox("Enable GitHub Agent Mode", value=st.session_state.github_mode)
            st.session_state.gh_token = st.text_input("GitHub Token", type="password", value=st.session_state.get("gh_token", ""))
            st.session_state.gh_repo = st.text_input("Repo (user/repo)", value=st.session_state.get("gh_repo", ""))

        sorted_chats = sorted(
            st.session_state.all_chats.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )

        for chat_id, chat_data in sorted_chats:
            is_active = (chat_id == st.session_state.active_chat_id)
            title = chat_data.get("title", "Untitled Chat")

            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                btn_type = "primary" if is_active else "secondary"
                if st.button(title, key=f"btn_{chat_id}", type=btn_type, use_container_width=True):
                    switch_chat(chat_id)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()

    active_chat = st.session_state.all_chats[st.session_state.active_chat_id]
    messages = active_chat.get("messages", [])

    if not messages:
        st.markdown("<h1 class='gemini-title'>Hello, Ameer</h1>", unsafe_allow_html=True)
        st.markdown("<p class='gemini-subtitle'>How can I help you code today?</p>", unsafe_allow_html=True)

        cols = st.columns(2)
        for idx, sug in enumerate(SUGGESTIONS):
            col = cols[idx % 2]
            with col:
                st.markdown(
                    f"""
                    <div class="suggestion-card">
                        <div style="font-size: 1.4rem; margin-bottom: 6px;">{sug['icon']}</div>
                        <div style="font-weight: 600; color: #0f172a; font-size: 0.95rem;">{sug['title']}</div>
                        <div style="color: #64748b; font-size: 0.82rem; margin-top: 2px;">{sug['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Use: {sug['title']}", key=f"sug_{idx}", use_container_width=True):
                    if sug['title'] == "GitHub Agent":
                        st.session_state.github_mode = True
                    submit_message(f"{sug['title']}: {sug['desc']}", selected_model)
                    st.rerun()
                st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # Render Conversation Loop
    for idx, msg in enumerate(messages):
        time_str = format_timestamp(msg.get("timestamp"))
        role = msg.get("role")
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                if time_str:
                    st.markdown(f"<div class='timestamp-badge'>{time_str}</div>", unsafe_allow_html=True)
                st.write(content)
        else:
            with st.chat_message("assistant", avatar="✨"):
                if time_str:
                    st.markdown(f"<div class='timestamp-badge'>{time_str}</div>", unsafe_allow_html=True)
                
                if thinking:
                    with st.expander("🧠 Model Thinking Process", expanded=False):
                        st.markdown(f"```text\n{thinking}\n```")

                st.markdown(content)
                render_copy_button(content, idx)

    # Attachments & Clear Button Section
    with st.expander("📎 Attach Files (PDF, Docs, Images, Zip - Max 500MB)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload files",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.active_chat_id}",
            max_upload_size=MAX_FILE_SIZE_MB,
            label_visibility="collapsed",
        )
        if uploaded_files:
            update_active_chat_file_context(uploaded_files)

        if active_chat.get("file_summary"):
            col_info, col_clear = st.columns([0.85, 0.15])
            with col_info:
                st.info(f"📎 {active_chat['file_summary']}")
            with col_clear:
                if st.button("❌ Clear Attached Files", key=f"clear_files_{st.session_state.active_chat_id}"):
                    clear_active_chat_file_context()
                    st.rerun()

    if st.session_state.get("github_mode", False):
        st.caption("🐙 **GitHub Agent Mode Active** — The model will inspect repo files and generate multi-file modifications.")

    prompt = st.chat_input("Ask Anything or request code changes across files...")
    if prompt:
        submit_message(prompt, selected_model)
        st.rerun()


if __name__ == "__main__":
    main()