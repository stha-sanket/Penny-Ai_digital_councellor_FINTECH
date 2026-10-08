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

import time
import uuid
import requests
from flask import Flask, request, send_from_directory, Response, jsonify
from flask_cors import CORS

from eval.tracker import (
    log_chat_interaction,
    update_trace_response,
    get_trace,
    get_traces,
    get_statistics,
    update_human_feedback,
    export_traces_data
)
from eval.evaluator import (
    evaluate_trace,
    evaluate_all_pending
)

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
        clean = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', ' ', clean)    # keep alphanumeric and Devanagari
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

    def search_with_scores(self, query: str, top_k: int = 3) -> list:
        """Find the top-K most relevant pages for query and return [(score, page)]."""
        if not self.pages:
            return []

        query_tokens = self._extract_keywords(query)
        if not query_tokens:
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
        return scored[:top_k]

    def search(self, query: str, top_k: int = 3) -> list:
        """Find the top-K most relevant pages for the query."""
        scored = self.search_with_scores(query, top_k)
        return [page for _, page in scored]


# ─── Initialize the page index on startup ───
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
page_index = PageIndex(DATA_DIR)


# ═══════════════════════════════════════════════════════════════
#  LANGUAGE PERSISTENCE & MANAGEMENT (language.txt)
#  1 = English, 0 = Nepali
# ═══════════════════════════════════════════════════════════════

LANGUAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'language.txt')

def get_current_language() -> str:
    """Returns '1' for English, '0' for Nepali. Defaults to '1'."""
    if os.path.exists(LANGUAGE_FILE):
        try:
            with open(LANGUAGE_FILE, 'r', encoding='utf-8') as f:
                val = f.read().strip()
                if val in ('0', '1'):
                    return val
        except Exception as e:
            print(f"⚠️ Error reading language.txt: {e}")
    return '1'

def set_current_language(val: str):
    """Stores '1' (English) or '0' (Nepali) into language.txt."""
    if val not in ('0', '1'):
        return
    try:
        with open(LANGUAGE_FILE, 'w', encoding='utf-8') as f:
            f.write(val)
        print(f"🌐 language.txt set to: {val} ({'English' if val == '1' else 'Nepali'})")
    except Exception as e:
        print(f"⚠️ Error writing to language.txt: {e}")

# Ensure language.txt exists on startup
if not os.path.exists(LANGUAGE_FILE):
    set_current_language('1')


def detect_language_command(text: str) -> tuple[str | None, bool]:
    """
    Detects if the user asks to switch language.
    Returns (lang_code, is_pure_switch_command):
      - lang_code: '1' for English, '0' for Nepali, or None
      - is_pure_switch_command: True if message was only asking to change language
    """
    clean = text.lower().strip()
    clean_no_punct = re.sub(r'^[^\w]+|[^\w]+$', '', clean)

    if clean_no_punct in ('english', 'angreji', 'अंग्रेजी'):
        return '1', True
    if clean_no_punct in ('nepali', 'नेपाली'):
        return '0', True

    nepali_patterns = [
        r'\b(?:speak|talk|reply|answer|converse|switch(?:\s+to)?|change(?:\s+language)?(?:\s+to)?|use)\s+(?:in\s+)?nepali\b',
        r'\b(?:can\s+you\s+|please\s+)?(?:speak|talk)\s+(?:in\s+)?nepali\b',
        r'\bnepali\s+(?:please|language)\b',
        r'\bnepali\s+ma\s+(?:bol|bola|bolnu|kura\s+gara|kura\s+garnus|jawab\s+deu)\b',
        r'नेपालीमा\s*(?:बोल|कुरा\s*गर|जवाफ\s*देउ|भन)',
        r'नेपाली\s*(?:बोल|भाषामा\s*बोल)',
    ]

    english_patterns = [
        r'\b(?:speak|talk|reply|answer|converse|switch(?:\s+to)?|change(?:\s+language)?(?:\s+to)?|use)\s+(?:in\s+)?(?:english|angreji)\b',
        r'\b(?:can\s+you\s+|please\s+)?(?:speak|talk)\s+(?:in\s+)?(?:english|angreji)\b',
        r'\b(?:english|angreji)\s+(?:please|language)\b',
        r'\b(?:english|angreji)\s+ma\s+(?:bol|bola|bolnu|kura\s+gara|kura\s+garnus|jawab\s+deu)\b',
        r'अंग्रेजीमा\s*(?:बोल|कुरा\s*गर|जवाफ\s*देउ|भन)',
        r'अंग्रेजी\s*(?:बोल|भाषामा\s*बोल)',
    ]

    def check_purity(phrase, pat):
        remainder = re.sub(pat, '', phrase, flags=re.I).strip()
        remainder = re.sub(r'\b(can you|could you|please|from now on|now|hai|na|kripaya|la|ok|okay)\b', '', remainder, flags=re.I).strip(' ,.!?')
        return len(remainder) == 0

    for pat in nepali_patterns:
        if re.search(pat, clean):
            return '0', check_purity(clean, pat)

    for pat in english_patterns:
        if re.search(pat, clean):
            return '1', check_purity(clean, pat)

    return None, False


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

SYSTEM_BASE_EN = """You are Penny, a friendly and concise AI assistant for Sunway College Kathmandu. Keep responses short (2-3 sentences max) and conversational. Be warm and helpful. Do not use markdown formatting, emojis, or special characters. Speak naturally as if in a real conversation.

CRITICAL LANGUAGE RULE: You MUST answer ONLY in English. Do NOT answer in Nepali or Devanagari script. Every response must be clear, natural English."""

CONTEXT_INSTRUCTION_EN = """
IMPORTANT KNOWLEDGE RULE:
You MUST answer ONLY using the CONTEXT provided below. Do NOT use any outside knowledge.
If the answer is NOT found in the context, say: "I don't have information about that in my knowledge base. I can only help with questions about Sunway College Kathmandu, its programs, staff, RAIN incubation center, and related topics."

CONTEXT:
{context}
"""

SYSTEM_BASE_NE = """You are Penny, a friendly and concise AI assistant for Sunway College Kathmandu. Keep responses short (2-3 sentences max) and conversational. Be warm and helpful. Do not use markdown formatting, emojis, or special characters. Speak naturally as if in a real conversation.

CRITICAL LANGUAGE RULE: You MUST reply ONLY in Nepali using Devanagari script (नेपाली भाषा / देवनागरी लिपि). NEVER reply in English. NEVER write romanized Nepali (no Latin script).
Even if the user writes in English or Roman Nepali, your entire response MUST be in pure Nepali Devanagari.

Examples:
User: timro nam k ho → Reply: मेरो नाम पेनी हो। म सनवे कलेजको AI सहायक हुँ।
User: kasto cha → Reply: म ठिक छु, धन्यवाद! तिमीलाई कसरी मद्दत गर्न सक्छु?
User: What courses do you have? → Reply: सनवे कलेजमा बीएससी आईटी, डेटा साइन्स र अन्य कम्प्युटिङ प्रोग्रामहरू उपलब्ध छन्।

NEVER write like this: "Mero naam Penny ho" — this is WRONG.
ALWAYS write like this: "मेरो नाम पेनी हो" — this is CORRECT."""

CONTEXT_INSTRUCTION_NE = """
IMPORTANT KNOWLEDGE RULE:
You MUST answer ONLY using the CONTEXT provided below. Do NOT use any outside knowledge.
Your response MUST be in Nepali in Devanagari script.
If the answer is NOT found in the context, say in Nepali: "मसँग मेरो ज्ञानकोषमा यस बारे जानकारी छैन। म केवल सनवे कलेज काठमाडौंका कार्यक्रम, कर्मचारी र RAIN इन्क्युबेशन सेन्टर सम्बन्धी प्रश्नहरूमा मद्दत गर्न सक्छु।"

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


def build_system_prompt(user_message: str, lang_code: str) -> str:
    """Build the language-specific system prompt, injecting relevant page context if available."""
    is_en = (lang_code == '1')
    base = SYSTEM_BASE_EN if is_en else SYSTEM_BASE_NE
    context_tpl = CONTEXT_INSTRUCTION_EN if is_en else CONTEXT_INSTRUCTION_NE

    if is_greeting(user_message):
        return base

    results = page_index.search(user_message, top_k=3)
    if not results:
        context = "(No relevant information found in the knowledge base.)"
    else:
        context_parts = []
        for page in results:
            context_parts.append(
                f"--- Source: {page['source']} | Section: {page['title']} ---\n"
                f"{page['content']}"
            )
        context = "\n\n".join(context_parts)

    return base + context_tpl.format(context=context)


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


# ─── Language state endpoint (GET / POST) ───
@app.route('/api/language', methods=['GET', 'POST'])
def api_language():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        val = str(data.get('language', '')).strip()
        if not val and 'lang' in data:
            val = '0' if data['lang'] == 'ne' else '1'
        if val in ('0', '1'):
            set_current_language(val)
            return jsonify({
                'ok': True,
                'language': val,
                'code': 'en' if val == '1' else 'ne',
                'name': 'English' if val == '1' else 'Nepali'
            })
        return jsonify({'error': 'Invalid language code. Use "1" for English or "0" for Nepali.'}), 400

    val = get_current_language()
    return jsonify({
        'language': val,
        'code': 'en' if val == '1' else 'ne',
        'name': 'English' if val == '1' else 'Nepali'
    })


# ─── NEW: Chat endpoint with page-index RAG (Supports streaming & non-streaming) ───
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    stream_requested = bool(data.get('stream', False))
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # 1. Check if the user is giving a command to change language
    detected_lang, is_pure_switch = detect_language_command(user_message)
    if detected_lang is not None:
        set_current_language(detected_lang)

    # 2. Get active language ('1' = English, '0' = Nepali)
    active_lang_code = get_current_language()
    lang_iso = 'en' if active_lang_code == '1' else 'ne'

    # If the user solely commanded to switch language, give an immediate conversational confirmation
    if detected_lang is not None and is_pure_switch:
        if active_lang_code == '1':
            confirmation = "Sure, I will speak in English from now on. How can I help you today?"
        else:
            confirmation = "हुन्छ, अबदेखि म नेपालीमा बोल्नेछु। म तपाईंलाई कसरी मद्दत गर्न सक्छु?"

        if stream_requested:
            def switch_stream():
                yield f"data: {json.dumps({'type': 'start', 'lang': lang_iso, 'language': active_lang_code})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'content': confirmation})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'response': confirmation, 'lang': lang_iso, 'language': active_lang_code})}\n\n"
            return Response(switch_stream(), mimetype='text/event-stream', headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
            })

        return jsonify({
            'response': confirmation,
            'lang': lang_iso,
            'language': active_lang_code
        })

    # 3. Build context-aware system prompt for the active language
    system_prompt = build_system_prompt(user_message, active_lang_code)

    ollama_payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ],
        'think': False,
        'stream': stream_requested,
        'keep_alive': -1,  # Keep model in GPU VRAM permanently (eliminates reload delay)
        'options': {
            'temperature': 0.7,
            'num_predict': 180,
            'num_ctx': 2048,
        }
    }

    if stream_requested:
        def stream_chat():
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'lang': lang_iso, 'language': active_lang_code})}\n\n"
            full_response_parts = []
            try:
                res = requests.post(OLLAMA_URL, json=ollama_payload, stream=True, timeout=60)
                res.raise_for_status()
                for line in res.iter_lines():
                    if not line:
                        continue
                    try:
                        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                        chunk = json.loads(line_str)
                    except Exception:
                        continue

                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        full_response_parts.append(content)
                        yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

                    if chunk.get('done', False):
                        break

                full_text = "".join(full_response_parts).strip()
                # Clean any lingering think/html tags
                full_text = re.sub(r'<think>[\s\S]*?</think>', '', full_text).strip()
                full_text = re.sub(r'<[^>]+>', '', full_text).strip()
                if not full_text:
                    full_text = "I didn't quite get that." if active_lang_code == '1' else "मैले बुझिन, कृपया फेरि भन्नुहोस्।"
                yield f"data: {json.dumps({'type': 'done', 'response': full_text, 'lang': lang_iso, 'language': active_lang_code})}\n\n"

            except Exception as e:
                print(f"Ollama stream error: {e}")
                err_msg = "Cannot reach Ollama. Make sure it's running." if "ConnectionError" in str(e) else f"Error: {e}"
                yield f"data: {json.dumps({'type': 'error', 'error': err_msg})}\n\n"

        return Response(stream_chat(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        })

    # Non-streaming fallback
    try:
        res = requests.post(OLLAMA_URL, json=ollama_payload, timeout=60)
        res.raise_for_status()
        result = res.json()

        response_text = result.get('message', {}).get('content', "I didn't quite get that.")
        response_text = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
        response_text = re.sub(r'<[^>]+>', '', response_text).strip()
        response_text = re.sub(r'\s+', ' ', response_text).strip()

        if not response_text:
            response_text = "I didn't quite get that." if active_lang_code == '1' else "मैले बुझिन, कृपया फेरि भन्नुहोस्।"

        return jsonify({
            'response': response_text,
            'lang': lang_iso,
            'language': active_lang_code
        })

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
    if not text:
        return Response('No text provided', status=400)

    # Determine language from request or language.txt
    lang = data.get('lang')
    if not lang:
        lang = 'en' if get_current_language() == '1' else 'ne'

    # Always ensure text with Devanagari script uses Nepali voice
    if re.search(r'[\u0900-\u097F]', text):
        lang = 'ne'

    try:
        piper_audio = generate_piper_tts(text, lang)
        if not piper_audio:
            # Fallback to alternate voice if primary isn't loaded
            fallback_lang = 'en' if lang != 'en' else 'ne'
            piper_audio = generate_piper_tts(text, fallback_lang)
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
