"""
Penny Voice Server
- Serves static files (index.html)
- /api/tts — converts text to natural speech using Microsoft Edge neural voices
"""

import asyncio
import io
import edge_tts
from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Voice configurations
VOICES = {
    'en': { 'voice': 'en-US-JennyNeural', 'rate': '-5%', 'pitch': '+0Hz' },
    'ne': { 'voice': 'ne-NP-HemkalaNeural', 'rate': '+0%', 'pitch': '+0Hz' },
}


async def generate_tts(text: str, lang: str = 'en') -> bytes:
    """Generate speech audio from text using Edge TTS."""
    config = VOICES.get(lang, VOICES['en'])
    communicate = edge_tts.Communicate(text, config['voice'], rate=config['rate'], pitch=config['pitch'])
    audio_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.write(chunk["data"])
    return audio_data.getvalue()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/chat.html')
def chat():
    return send_from_directory('.', 'chat.html')


@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text', '')
    lang = data.get('lang', 'en')
    if not text:
        return Response('No text provided', status=400)

    try:
        audio_bytes = asyncio.run(generate_tts(text, lang))
        return Response(audio_bytes, mimetype='audio/mpeg', headers={
            'Content-Type': 'audio/mpeg',
            'Cache-Control': 'no-cache'
        })
    except Exception as e:
        print(f"TTS Error: {e}")
        return Response(f'TTS failed: {str(e)}', status=500)


if __name__ == '__main__':
    print("🤖 Penny Voice Server running at http://localhost:5000")
    print(f"   Voices: EN={VOICES['en']['voice']}, NE={VOICES['ne']['voice']}")
    app.run(host='0.0.0.0', port=5000, debug=False)
