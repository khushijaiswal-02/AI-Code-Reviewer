
#groq_client.py (file kept as grok_client.py for import compatibility)

#A small wrapper around the Groq API.
"""
WHY THIS FILE EXISTS:
Keeping the API call in its own module means:
- The API key is loaded from .env in ONE place
- If you ever switch models or providers, you only change this file
- app.py stays focused on UI logic, not HTTP details

Groq's API is OpenAI-compatible, so we use the official `openai` Python
library pointed at Groq's base URL. This is the standard, well-supported
way to call Groq.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (GROQ_API_KEY=...)
import streamlit as st
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Chat model used for code reviews and confidence scoring.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Speech-to-text model used for the voice input feature.
GROQ_STT_MODEL = "whisper-large-v3-turbo"


def _get_client() -> OpenAI:
    
    #Create and return an OpenAI-compatible client pointed at Groq.

    #Raises ValueError = If GROQ_API_KEY is not found in environment variables.
    
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure your .env file exists "
            "and contains: GROQ_API_KEY=your_key_here"
        )

    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )


def get_code_review(prompt: str) -> str:
    
    #Send the final RAG prompt to Groq and return the model's response text(non-streaming).

    #Parameters = prompt : str
        #The complete prompt (retrieved context + user code), built by rag_pipeline.build_review_prompt() / build_review_prompt_with_score().

    #Returns str = The model's response text (the formatted code review).
    
    client = _get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # Lower temperature = more consistent, factual reviews
    )

    return response.choices[0].message.content


def stream_code_review(prompt: str):
    
    #Send the final RAG prompt to Groq and yield the response in chunks as they arrive (streaming).
    #WHY STREAMING:For long reviews, waiting several seconds for the full response feels slow. Streaming displays text as it's generated, similar to ChatGPT,giving immediate feedback that the system is working.
    #USAGE (in Streamlit): st.write_stream(stream_code_review(prompt))
    #Yields str = Successive text chunks from the model's response.

    client = _get_client()

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def transcribe_audio(audio_bytes: bytes) -> str:
    
    #Send recorded audio to Groq's Whisper-based Speech-to-Text API and return the transcribed text.

    #WHY THE OPENAI SDK HERE (unlike the older xAI version):Groq exposes audio transcription 
    # as a standard OpenAI-compatible endpoint (client.audio.transcriptions.create), 
    # so we use the same `openai` client as the chat calls - no raw `requests` needed.

    # Takes raw WAV audio bytes from the microphone
    # Sends to Groq's Whisper model (speech-to-text)
    # Returns the transcribed text string
    # That text gets inserted into the code input box

    client = _get_client()

    # The Groq SDK expects a file-like object with a name attribute
    # (it uses the extension to determine the audio format).
    import io
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"

    transcript = client.audio.transcriptions.create(
        model=GROQ_STT_MODEL,
        file=audio_file,
        language="en",
        response_format="text",
    )

    # When response_format="text", the SDK may return either a plain
    # string or an object with a `.text` attribute depending on version.
    if isinstance(transcript, str):
        return transcript.strip()
    return getattr(transcript, "text", "").strip()


def get_confidence_score(prompt: str, review_result: str) -> float:
    
    #Ask Groq to self-rate its confidence in the review it just gave,
    #based on whether the retrieved context was sufficient.

    
    try:
        client = _get_client()
    except ValueError:
        return 50.0

    rating_prompt = f"""
Based on the following code review and the context that was available to you,
rate your confidence that this review is accurate and well-grounded in the
provided reference material, on a scale of 0 to 100.

Respond with ONLY a number (e.g., "85"). No explanation, no extra text.

Review:
{review_result}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": rating_prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Extract first number found in the response, in case the model
        # adds extra text despite instructions.
        import re
        match = re.search(r"\d+(\.\d+)?", raw)
        if match:
            score = float(match.group())
            return max(0.0, min(100.0, score))  # clamp to 0-100
    except Exception:
        pass

    return 50.0  # fallback default if anything goes wrong