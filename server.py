"""
Penny Voice Server
- Serves static files (index.html, chat.html)
- /api/tts  — converts text to natural speech using Microsoft Edge neural voices
- /api/chat — page-index RAG: answers ONLY from data/ folder context
"""

import io
import os
import re
import math
import glob
import json
import queue
import wave
from collections import Counter

import requests
from flask import Flask, request, send_from_directory, Response, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ─── Ollama Config ───
OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'gemma4:e2b'

# ─── Piper Local ONNX Voice Models ───
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'piper')
PIPER_VOICES = {}
try:
    from piper.voice import PiperVoice
    en_onnx = os.path.join(MODELS_DIR, 'en_US-lessac-medium.onnx')
    en_json = os.path.join(MODELS_DIR, 'en_US-lessac-medium.onnx.json')
    ne_onnx = os.path.join(MODELS_DIR, 'ne_NP-google-medium.onnx')
    ne_json = os.path.join(MODELS_DIR, 'ne_NP-google-medium.onnx.json')

    if os.path.exists(en_onnx) and os.path.exists(en_json):
        PIPER_VOICES['en'] = PiperVoice.load(en_onnx, config_path=en_json)
        print("🔊 Piper ONNX English voice loaded (en_US-lessac-medium)")
    if os.path.exists(ne_onnx) and os.path.exists(ne_json):
        PIPER_VOICES['ne'] = PiperVoice.load(ne_onnx, config_path=ne_json)
        print("🔊 Piper ONNX Nepali voice loaded (ne_NP-google-medium)")
except Exception as e:
    print(f"⚠️ Piper ONNX voice loading warning: {e}")


# ═══════════════════════════════════════════════════════════════
#  PAGE INDEX — loads, chunks, and keyword-indexes data/ files
# ═══════════════════════════════════════════════════════════════

class PageIndex:
    """
    Lightweight keyword-based page index for markdown files.

    On init:
      - Reads every .md file in data_dir
      - Splits into "pages" (sections by ## headers or --- separators)
      - Builds a TF-IDF-like keyword index for fast retrieval

    search(query, top_k) returns the most relevant pages as context.
    """

    # Common English stop words to ignore during indexing
    STOP_WORDS = frozenset([
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or', 'if',
        'while', 'about', 'up', 'that', 'this', 'it', 'its', 'he', 'she',
        'they', 'them', 'we', 'you', 'i', 'me', 'my', 'your', 'his', 'her',
        'our', 'their', 'what', 'which', 'who', 'whom', 'these', 'those',
    ])

    def __init__(self, data_dir: str):
        self.pages = []          # list of {source, title, content, keywords}
        self.idf = {}            # term → inverse document frequency
        self._load_and_index(data_dir)

    # ─── Load & split markdown files into pages ───
    def _load_and_index(self, data_dir: str):
        md_files = sorted(glob.glob(os.path.join(data_dir, '*.md')))
        if not md_files:
            print(f"⚠️  No .md files found in {data_dir}")
            return

        for filepath in md_files:
            filename = os.path.basename(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = f.read()
            pages = self._split_into_pages(raw, filename)
            self.pages.extend(pages)

        # Build IDF across all pages
        self._build_idf()
        print(f"📄 Page Index loaded: {len(self.pages)} pages from {len(md_files)} files")
        for p in self.pages:
            print(f"   • [{p['source']}] {p['title']}")

    def _split_into_pages(self, text: str, source: str) -> list:
        """Split markdown into pages by ## headers or --- separators."""
        pages = []
        # Split by ## headers or --- horizontal rules
        sections = re.split(r'\n(?=#{1,3}\s)|(?:\n---\n)', text)

        for section in sections:
            section = section.strip()
            if not section or len(section) < 10:
                continue

            # Extract title from first heading line
            title_match = re.match(r'^#{1,5}\s+(.+)', section)
            title = title_match.group(1).strip() if title_match else source

            keywords = self._extract_keywords(section)
            pages.append({
                'source': source,
                'title': title,
                'content': section,
                'keywords': keywords,
            })

        return pages

    def _extract_keywords(self, text: str) -> Counter:
        """Extract and count meaningful keywords from text."""
        # Remove markdown formatting
        clean = re.sub(r'[#*_\[\]()>`|~\-]', ' ', text)
        clean = re.sub(r'https?://\S+', '', clean)       # remove URLs
        clean = re.sub(r'\S+@\S+', '', clean)             # remove emails
        clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean)    # keep alphanumeric
        tokens = clean.lower().split()
        # Filter stop words and very short tokens
        meaningful = [t for t in tokens if t not in self.STOP_WORDS and len(t) > 1]
        return Counter(meaningful)

    def _build_idf(self):
        """Compute inverse document frequency for each term."""
        n = len(self.pages)
        if n == 0:
            return
        doc_freq = Counter()
        for page in self.pages:
            doc_freq.update(page['keywords'].keys())
        self.idf = {
            term: math.log((n + 1) / (freq + 1)) + 1
            for term, freq in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 3) -> list:
        """Find the top-K most relevant pages for the query."""
        if not self.pages:
            return []

        query_tokens = self._extract_keywords(query)
        if not query_tokens:
            # If no meaningful keywords, return all pages (for greetings etc.)
            return []

        scored = []
        for page in self.pages:
            score = 0.0
            for term, q_count in query_tokens.items():
                if term in page['keywords']:
                    tf = page['keywords'][term]
                    idf = self.idf.get(term, 1.0)
                    score += tf * idf * q_count
            if score > 0:
                scored.append((score, page))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [page for _, page in scored[:top_k]]


# ─── Initialize the page index on startup ───
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
page_index = PageIndex(DATA_DIR)


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

SYSTEM_BASE = """You are Penny, a friendly and concise AI assistant for Sunway College Kathmandu. Keep responses short (2-3 sentences max) and conversational. Be warm and helpful. Do not use markdown formatting, emojis, or special characters. Speak naturally as if in a real conversation.

CRITICAL LANGUAGE RULE: When the user writes in Nepali (even in Roman script), you MUST reply ONLY in Devanagari script. NEVER write romanized Nepali.

Examples:
User: timro nam k ho → Reply: मेरो नाम पेनी हो। म सनवे कलेजको AI सहायक हुँ।
User: kasto cha → Reply: म ठिक छु, धन्यवाद! तिमीलाई कसरी मद्दत गर्न सक्छु?

NEVER write like this: "Mero naam Penny ho" — this is WRONG.
ALWAYS write like this: "मेरो नाम पेनी हो" — this is CORRECT."""

CONTEXT_INSTRUCTION = """
IMPORTANT KNOWLEDGE RULE:
You MUST answer ONLY using the CONTEXT provided below. Do NOT use any outside knowledge.
If the answer is NOT found in the context, say: "I don't have information about that in my knowledge base. I can only help with questions about Sunway College Kathmandu, its programs, staff, RAIN incubation center, and related topics."

CONTEXT:
{context}
"""

GREETING_WORDS = frozenset([
    'hi', 'hello', 'hey', 'namaste', 'namaskar', 'greetings', 'good morning',
    'good afternoon', 'good evening', 'sup', 'yo', 'howdy', 'thanks',
    'thank you', 'bye', 'goodbye', 'see you', 'ok', 'okay',
])


def is_greeting(text: str) -> bool:
    """Check if the message is a simple greeting/farewell."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return cleaned in GREETING_WORDS or len(cleaned.split()) <= 2 and any(
        w in cleaned.split() for w in GREETING_WORDS
    )


def build_system_prompt(user_message: str) -> str:
    """Build the system prompt, injecting relevant page context if available."""
    if is_greeting(user_message):
        return SYSTEM_BASE

    results = page_index.search(user_message, top_k=3)
    if not results:
        # No relevant pages found — still constrain the LLM
        context = "(No relevant information found in the knowledge base.)"
    else:
        context_parts = []
        for page in results:
            context_parts.append(
                f"--- Source: {page['source']} | Section: {page['title']} ---\n"
                f"{page['content']}"
            )
        context = "\n\n".join(context_parts)

    return SYSTEM_BASE + CONTEXT_INSTRUCTION.format(context=context)


# ═══════════════════════════════════════════════════════════════
#  TTS — Piper Local ONNX (Primary) + Edge TTS (Fallback)
# ═══════════════════════════════════════════════════════════════

def generate_piper_tts(text: str, lang: str = 'en') -> bytes | None:
    """Generate ultra-fast speech audio locally using Piper ONNX."""
    voice = PIPER_VOICES.get(lang) or PIPER_VOICES.get('en')
    if not voice:
        return None
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        voice.synthesize_wav(text, wav_file)
    return buf.getvalue()



# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/chat.html')
def chat():
    return send_from_directory('.', 'chat.html')


# ─── Multi-Device State Synchronization ───
current_state = 'idle'
state_subscribers = []

@app.route('/api/state', methods=['GET', 'POST'])
def handle_state():
    global current_state
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        new_state = data.get('state', 'idle')
        current_state = new_state
        dead = []
        for q in state_subscribers:
            try:
                q.put_nowait(current_state)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in state_subscribers:
                state_subscribers.remove(q)
        return jsonify({'ok': True, 'state': current_state})
    return jsonify({'state': current_state})


@app.route('/api/events')
def events():
    """Server-Sent Events (SSE) stream to sync robot face state across laptops."""
    def stream():
        q = queue.Queue(maxsize=20)
        state_subscribers.append(q)
        # Send current state immediately on connection
        yield f"data: {json.dumps({'state': current_state})}\n\n"
        try:
            while True:
                try:
                    state = q.get(timeout=20)
                    yield f"data: {json.dumps({'state': state})}\n\n"
                except queue.Empty:
                    # Heartbeat to keep connection alive
                    yield ": ping\n\n"
        except GeneratorExit:
            if q in state_subscribers:
                state_subscribers.remove(q)

    return Response(stream(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive',
    })


# ─── NEW: Chat endpoint with page-index RAG ───
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Build context-aware system prompt
    system_prompt = build_system_prompt(user_message)

    try:
        ollama_payload = {
            'model': MODEL,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            'think': False,
            'stream': False,
            'keep_alive': -1,  # Keep model in GPU VRAM permanently (eliminates 13s reload delay)
            'options': {
                'temperature': 0.7,
                'num_predict': 180,
                'num_ctx': 2048,
            }
        }

        res = requests.post(OLLAMA_URL, json=ollama_payload, timeout=60)
        res.raise_for_status()
        result = res.json()

        response_text = result.get('message', {}).get('content', "I didn't quite get that.")
        # Clean up any think tags or HTML
        response_text = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
        response_text = re.sub(r'<[^>]+>', '', response_text).strip()
        response_text = re.sub(r'\s+', ' ', response_text).strip()

        if not response_text:
            response_text = "I didn't quite get that."

        return jsonify({'response': response_text})

    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Cannot reach Ollama. Make sure it\'s running on localhost:11434'}), 503
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({'error': f'Something went wrong: {str(e)}'}), 500


# ─── TTS endpoint (100% Local Offline Piper ONNX) ───
@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    lang = data.get('lang', 'en')
    if not text:
        return Response('No text provided', status=400)

    try:
        piper_audio = generate_piper_tts(text, lang)
        if not piper_audio:
            return Response('No local voice model available', status=500)

        return Response(piper_audio, mimetype='audio/wav', headers={
            'Content-Type': 'audio/wav',
            'Cache-Control': 'no-cache'
        })
    except Exception as e:
        print(f"TTS Error: {e}")
        return Response(f'TTS failed: {str(e)}', status=500)


if __name__ == '__main__':
    loaded_piper = ', '.join(PIPER_VOICES.keys()) or 'None'
    print("🤖 Penny Voice Server running at http://localhost:5000")
    print(f"   Local ONNX Voices: {loaded_piper}")
    print(f"   Data dir: {DATA_DIR}")
    app.run(host='0.0.0.0', port=5000, debug=False)
