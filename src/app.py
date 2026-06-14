"""
app.py

Main Streamlit application for the AI Code Reviewer (RAG-based).

WORKFLOW:
1. User pastes their source code into a text area.
2. On clicking "Review Code", the app:
   a. Retrieves relevant guideline chunks from the FAISS knowledge base
      (Clean Code Guidelines, OWASP security docs, Java best practices).
   b. Builds a final prompt combining those guidelines + the user's code.
   c. Sends the prompt to Grok.
   d. Displays the structured review.

WHY @st.cache_resource:
Loading the embedding model and FAISS index takes a few seconds. Without
caching, Streamlit would reload them on EVERY button click or page
interaction, making the app feel slow. @st.cache_resource loads it once
per session and reuses it.
"""

import streamlit as st
from streamlit_mic_recorder import mic_recorder
from rag_pipeline import load_vector_db, build_review_prompt_with_score
from grok_client import stream_code_review, transcribe_audio


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Code Reviewer",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for a dark, IDE-style look
st.markdown(
    """
    <style>
    /* Overall font for body text */
    html, body, [class*="css"] {
        font-family: 'Consolas', 'Monaco', monospace;
    }

    /* Title block */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00d4aa;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.8rem;
        font-family: 'Consolas', 'Monaco', monospace;
    }

    /* Code input area styled like an editor */
    .stTextArea textarea {
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-size: 0.92rem !important;
        background-color: #161a23 !important;
        color: #d4d4d4 !important;
        border: 1px solid #2d333d !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #00d4aa !important;
        box-shadow: 0 0 0 1px #00d4aa !important;
    }

    /* Primary button */
    .stButton > button {
        background-color: #00d4aa;
        color: #0e1117;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    .stButton > button:hover {
        background-color: #00b894;
        color: #0e1117;
    }

    /* Result container card */
    .result-card {
        background-color: #161a23;
        border: 1px solid #2d333d;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00d4aa;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #2d333d;
        padding-bottom: 0.5rem;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h3 {
        color: #00d4aa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# DYNAMIC BACKGROUND COLOR
# ---------------------------------------------------------------------------
# We store the chosen background color in session_state so it persists
# across interactions (button clicks, etc.) within the same session.
# This CSS block is injected SEPARATELY and AFTER the static CSS above,
# so it can override the default background color.
if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#0e1117"  # default dark background

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {st.session_state.bg_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# LOAD RAG RETRIEVER (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vector_db():
    return load_vector_db()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.write(
        "This AI Code Reviewer uses Retrieval-Augmented Generation (RAG) "
        "to ground its feedback in established standards, including "
        "Clean Code Guidelines, OWASP security practices, and Java "
        "best practices."
    )
    st.markdown("### How it works")
    st.write(
        "1. Paste your code below\n"
        "2. The system retrieves the most relevant guidelines\n"
        "3. Grok analyzes your code using those guidelines\n"
        "4. You receive a structured review with severity levels and fixes"
    )
    st.markdown("---")
    st.caption("Supports Java, Python, C, C++, and JavaScript.")

    st.markdown("---")
    st.markdown("### Appearance")
    chosen_color = st.color_picker(
        "Background color",
        value=st.session_state.bg_color,
    )
    if chosen_color != st.session_state.bg_color:
        st.session_state.bg_color = chosen_color
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">&lt;/&gt; AI Code Reviewer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle"># Automated code review grounded in retrieved best-practice documentation (RAG)</div>',
    unsafe_allow_html=True,
)

st.markdown("##### Source Code Input")

# Initialize session state for the code input box.
if "code_input_value" not in st.session_state:
    st.session_state.code_input_value = ""

# --- Voice input ---
col_mic, col_label = st.columns([1, 6])
with col_mic:
    audio = mic_recorder(
        start_prompt="🎙️ Record",
        stop_prompt="⏹️ Stop",
        just_once=True,
        format="wav",
        key="recorder",
    )
with col_label:
    st.caption(
        "Use voice to describe your code or paste a short snippet aloud. "
        "Best for descriptions, not exact syntax."
    )

# Process audio BEFORE rendering the text_area so the updated value
# is already in session_state when the widget renders.
if audio and audio.get("bytes"):
    with st.spinner("Transcribing audio..."):
        try:
            transcribed_text = transcribe_audio(audio["bytes"])
            if transcribed_text:
                existing = st.session_state.code_input_value.strip()
                if existing:
                    st.session_state.code_input_value = existing + "\n" + transcribed_text
                else:
                    st.session_state.code_input_value = transcribed_text
                st.success(f"Transcribed: *{transcribed_text[:80]}{'...' if len(transcribed_text) > 80 else ''}*")
            else:
                st.warning("No speech detected. Please try again.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")

# The text_area reads its initial value from session_state on every render.
# When the user types manually, we sync it back via on_change callback.
def sync_text_area():
    st.session_state.code_input_value = st.session_state.code_input_widget

user_code = st.text_area(
    label="",
    height=320,
    placeholder="// Paste your code here or use the mic above...",
    label_visibility="collapsed",
    value=st.session_state.code_input_value,
    key="code_input_widget",
    on_change=sync_text_area,
)

review_clicked = st.button("▶  Review Code", type="primary")

if review_clicked:
    if not user_code.strip():
        st.warning("Please paste some code before requesting a review.")
    else:
        try:
            vector_db = get_vector_db()

            # Step 1: Detect language + retrieve guidelines + get relevance score
            with st.spinner("Detecting language and retrieving relevant guidelines..."):
                final_prompt, relevance_score, detected_language = build_review_prompt_with_score(
                    vector_db, user_code, k=3
                )

            # Show detected language as a badge
            lang_colors = {
                "Java":       "#f89820",
                "Python":     "#3572A5",
                "C":          "#555555",
                "C++":        "#f34b7d",
                "JavaScript": "#f1e05a",
                "Unknown":    "#888888",
            }
            badge_color = lang_colors.get(detected_language, "#888888")
            st.markdown(
                f'<span style="background:{badge_color};color:#0e1117;'
                f'padding:3px 12px;border-radius:12px;font-size:0.85rem;'
                f'font-weight:700;font-family:monospace;">'
                f'⟨/⟩ {detected_language}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Step 2: Stream the review from Groq
            st.markdown(
                '<div class="result-card"><div class="result-header">📋 Review Result</div>',
                unsafe_allow_html=True,
            )

            review_result = st.write_stream(stream_code_review(final_prompt))

            st.markdown('</div>', unsafe_allow_html=True)

            # Step 3: Show relevance score only (model confidence removed)
            st.markdown("---")
            st.markdown("##### Accuracy Indicator")

            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Retrieval Relevance", f"{relevance_score}%")
            with col2:
                st.caption(
                    "Retrieval Relevance reflects how well your knowledge base "
                    "matched the submitted code. Higher = more relevant guidelines "
                    "were found. This is a heuristic indicator, not a verified "
                    "correctness measure."
                )

        except ValueError as ve:
            st.error(f"Configuration error: {ve}")
        except Exception as e:
            st.error(
                "Something went wrong while generating the review. "
                f"Details: {e}"
            )