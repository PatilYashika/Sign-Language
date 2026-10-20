import streamlit as st
import streamlit.components.v1 as components

def speak_text(text: str, enabled: bool = True, rate: float = 1.0, pitch: float = 1.0, key_suffix: str = ""):
    """
    Synthesizes speech in real-time via the browser's native Web Speech API.
    Zero audio lag, high clarity, and works across Windows, Chrome, Edge, and Firefox.
    """
    if not enabled or not text:
        return

    safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'").replace("\n", " ").strip()
    if not safe_text:
        return

    # Unique script invocation
    html_code = f"""
    <div style="display:none;" id="tts_trigger_{key_suffix}">
    <script>
    (function() {{
        try {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance("{safe_text}");
                utterance.rate = {rate};
                utterance.pitch = {pitch};
                utterance.lang = 'en-US';
                window.speechSynthesis.speak(utterance);
            }}
        }} catch(e) {{
            console.error("TTS Error:", e);
        }}
    }})();
    </script>
    </div>
    """
    components.html(html_code, height=0, width=0)
