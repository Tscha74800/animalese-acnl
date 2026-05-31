import os
import io
import re
import numpy as np
from scipy.io import wavfile
import librosa
from flask import Flask, request, send_file, render_template_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR_BOY = os.path.join(BASE_DIR, "voices", "BOY")
VOICES_DIR_GIRL = os.path.join(BASE_DIR, "voices", "GIRL")

# Paramètres de rythme
SPEED_FAST = 1.75
SPEED_SLOW = 0.6
OVERLAP_FACTOR = 0.5
SPACE_DURATION_SECS = 0.015
PUNC_DURATION_SECS = 0.08


CHARACTERS = {
    "Tom Nook": {"pitch": 1.11, "gender": "boy"},
    "Amiral": {"pitch": 1.05, "gender": "boy"},
    "Sonny Resetti": {"pitch": 1.05, "gender": "boy"},
    "Don Resetti": {"pitch": 1.02, "gender": "boy"},
    "Le Maire": {"pitch": 1.00, "gender": "boy"},
    "Gulliver": {"pitch": 0.95, "gender": "boy"},
    "Albin": {"pitch": 1.12, "gender": "boy"},
    "Méli Mélo": {"pitch": 1.50, "gender": "boy"},
    "Élisabec": {"pitch": 1.1, "gender": "girl"},
    "Opélie": {"pitch": 1.15, "gender": "girl"},
    "Antoine": {"pitch": 1.18, "gender": "boy"},
    "Katrina": {"pitch": 0.93, "gender": "girl"},
    "Marie": {"pitch": 1.25, "gender": "girl"},
    "Risette": {"pitch": 1.3, "gender": "girl"},
    "Racine": {"pitch": 1.21, "gender": "boy"},
    "Carla": {"pitch": 0.95, "gender": "girl"},
    "Porcelette": {"pitch": 1.10, "gender": "girl"},
    "Rounard": {"pitch": 0.85, "gender": "boy"},
    "Ginette": {"pitch": 0.95, "gender": "girl"},
    "Lazare": {"pitch": 0.70, "gender": "boy"},
    "Charly": {"pitch": 1.10, "gender": "boy"},
    "Tortimer": {"pitch": 0.95, "gender": "boy"},
    "Coco": {"pitch": 1.00, "gender": "girl"},
    "Marito": {"pitch": 1.05, "gender": "boy"},
    "Bibi": {"pitch": 1.05, "gender": "girl"},
    "Mathéo": {"pitch": 1.15, "gender": "boy"},
    "Chavrina": {"pitch": 0.95, "gender": "girl"},
    "Théo": {"pitch": 1.10, "gender": "boy"},
    "Miro": {"pitch": 1.20, "gender": "boy"},
    "Shaki": {"pitch": 1.00, "gender": "girl"}
}


def load_phoneme_libraries(target_sr=44100):
    """ Charge séparément les dossiers BOY et GIRL en RAM """
    libraries = {'boy': {}, 'girl': {}}
    valid_extensions = ('.wav', '.ogg', '.mp3', '.flac')
    
    # Chargement Bibliothèque Garçons
    if os.path.exists(VOICES_DIR_BOY):
        for file in os.listdir(VOICES_DIR_BOY):
            if file.lower().endswith(valid_extensions):
                name_key = os.path.splitext(file)[0].lower().strip()
                try:
                    audio, _ = librosa.load(os.path.join(VOICES_DIR_BOY, file), sr=target_sr, mono=True)
                    libraries['boy'][name_key] = audio
                except: pass
    else:
        print(f"[ATTENTION] Dossier BOY introuvable : {VOICES_DIR_BOY}")

    # Chargement Bibliothèque Filles
    if os.path.exists(VOICES_DIR_GIRL):
        for file in os.listdir(VOICES_DIR_GIRL):
            if file.lower().endswith(valid_extensions):
                name_key = os.path.splitext(file)[0].lower().strip()
                try:
                    audio, _ = librosa.load(os.path.join(VOICES_DIR_GIRL, file), sr=target_sr, mono=True)
                    libraries['girl'][name_key] = audio
                except: pass
    else:
        print(f"[ATTENTION] Dossier GIRL introuvable : {VOICES_DIR_GIRL}")
        
    print(f"[SUCCÈS] Phonèmes chargés : {len(libraries['boy'])} Garçon | {len(libraries['girl'])} Fille")
    return libraries

def reduce_vowels(text):
    vowels = "aeiouyàâäéèêëîïôöùûü"
    words = re.split(r'(\W+)', text) 
    
    reduced = []
    for word in words:
        if not word.strip() or not any(c.isalpha() for c in word):
            reduced.append(word)
            continue
            
        vowel_count = 0
        new_word = []
        for char in word:
            if char.lower() in vowels:
                vowel_count += 1
                if vowel_count <= 2:
                    new_word.append(char)
            else:
                new_word.append(char)
                
        reduced.append("".join(new_word))
    return "".join(reduced)

def tokenize_text(text, available_keys):
    text = reduce_vowels(text.lower())
    tokens = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == ' ':
            tokens.append(('SPACE', ' '))
            i += 1
            continue
        elif char in ".,!?":
            tokens.append(('PUNC', char))
            i += 1
            continue
            
        matched = False
        for width in range(4, 0, -1):
            if i + width <= len(text):
                substring = text[i:i+width]
                if substring in available_keys:
                    tokens.append(('PHONEME', substring))
                    i += width
                    matched = True
                    break
                    
        if not matched:
            i += 1
    return tokens

def analyze_intonation_contexts(tokens):
    contexts = []
    current_mood = '.'
    for token_type, value in reversed(tokens):
        if token_type == 'PUNC': current_mood = value
        contexts.append(current_mood)
    contexts.reverse()
    return contexts

def get_dynamic_pitch(idx, total_tokens, mood, base_character_pitch):
    progress = idx / max(1, total_tokens - 1)
    
    macro_arc = 0.08 * np.sin(progress * np.pi) - 0.03
    micro_bounce = 0.06 * np.sin(idx * 2.5) + 0.03 * np.cos(idx * 1.2)
    
    end_arc = 0
    if progress > 0.85:
        end_progress = (progress - 0.85) / 0.15
        if mood == '?': end_arc = end_progress * 0.40
        elif mood == '!': end_arc = end_progress * 0.10
        else: end_arc = -(end_progress * 0.10)
            
    phrase_modifier = 1.0 + macro_arc + end_arc
    return (base_character_pitch * phrase_modifier) + micro_bounce

def generate_animalese_v31(text, library, base_pitch, speed_factor, sample_rate=44100):
    tokens = tokenize_text(text, library.keys())
    if not tokens: return np.zeros(100, dtype=np.int16)
        
    mood_contexts = analyze_intonation_contexts(tokens)
    audio_triggers = []
    current_sample_idx = 0
    
    space_samples = int(SPACE_DURATION_SECS * sample_rate)
    punc_samples = int(PUNC_DURATION_SECS * sample_rate)
    
    for idx, (token_type, value) in enumerate(tokens):
        if token_type == 'PHONEME':
            raw_audio = library[value]
            current_pitch = get_dynamic_pitch(idx, len(tokens), mood_contexts[idx], base_pitch)
            
            duration_samples = int(len(raw_audio) / max(0.1, speed_factor))
            if duration_samples <= 0: continue
            
            steps = (np.arange(duration_samples) * current_pitch).astype(np.int32) % len(raw_audio)
            processed_audio = raw_audio[steps].copy()
            
            fade_len = min(int(sample_rate * 0.005), len(processed_audio) // 2)
            if fade_len > 0:
                window = np.ones(len(processed_audio), dtype=np.float32)
                window[:fade_len] = np.linspace(0.0, 1.0, fade_len)
                window[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
                processed_audio *= window
                
            audio_triggers.append((current_sample_idx, processed_audio))
            current_sample_idx += int(len(processed_audio) * (1.0 - max(0.0, min(0.99, OVERLAP_FACTOR))))
            
        elif token_type == 'SPACE': current_sample_idx += space_samples
        elif token_type == 'PUNC': current_sample_idx += punc_samples

    end_samples = [pos + len(audio) for pos, audio in audio_triggers]
    if not end_samples: return np.zeros(100, dtype=np.int16)
    
    output_buffer = np.zeros(max(end_samples) + int(sample_rate * 0.05), dtype=np.float32)
    for start_pos, audio in audio_triggers:
        output_buffer[start_pos:start_pos + len(audio)] += audio
        
    max_val = np.max(np.abs(output_buffer))
    if max_val > 0: output_buffer = (output_buffer / max_val) * 0.85
        
    return (output_buffer * 32767).astype(np.int16)

def clean_filename_part(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\séèàùçâêîôûëïü]", "", text)
    words = text.split()
    return "_".join(words[:4])


app = Flask(__name__)

# Chargement global pour que Gunicorn (serveur de prod) initialise les voix au démarrage
print("--- CHARGEMENT DES BANQUES AUDIO ---")
phoneme_libs = load_phoneme_libraries(44100)
print("--- BANQUES CHARGÉES ---")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Animalese Studio V40 - Multi-Genres</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e2e; color: #cdd6f4; margin: 0; padding: 2rem; display: flex; justify-content: center; }
        .container { background-color: #313244; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); width: 100%; max-width: 600px; }
        h1 { text-align: center; color: #a6e3a1; margin-bottom: 1.5rem; }
        label { display: block; font-weight: bold; margin-bottom: 0.5rem; color: #89b4fa; }
        select, textarea { width: 100%; padding: 0.8rem; margin-bottom: 1.5rem; border: 1px solid #45475a; border-radius: 8px; background-color: #1e1e2e; color: #cdd6f4; font-size: 1rem; box-sizing: border-box; }
        textarea { resize: vertical; min-height: 100px; }
        .radio-group { display: flex; gap: 1.5rem; margin-bottom: 1.5rem; background-color: #1e1e2e; padding: 0.8rem; border-radius: 8px; border: 1px solid #45475a; }
        .radio-option { display: flex; align-items: center; gap: 0.5rem; cursor: pointer; }
        .radio-option input { cursor: pointer; accent-color: #a6e3a1; }
        button { width: 100%; padding: 1rem; background-color: #89b4fa; color: #11111b; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: background-color 0.2s; }
        button:hover { background-color: #74c7ec; }
        .result-section { margin-top: 2rem; display: none; flex-direction: column; gap: 1rem; align-items: center; background-color: #1e1e2e; padding: 1.5rem; border-radius: 8px; border: 1px solid #45475a; }
        audio { width: 100%; }
        .download-btn { display: inline-block; padding: 0.8rem 1.5rem; background-color: #a6e3a1; color: #11111b; text-decoration: none; border-radius: 8px; font-weight: bold; text-align: center; width: 100%; box-sizing: border-box; transition: background-color 0.2s; }
        .download-btn:hover { background-color: #94e2d5; }
        .loader { display: none; text-align: center; margin-top: 1rem; color: #f9e2af; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Animalese Studio V40</h1>
        
        <label for="character">Personnage :</label>
        <select id="character">
            {% for name, data in characters.items() %}
            <option value="{{ name }}">{{ name }} {% if data.gender == 'boy' %}👦{% else %}👧{% endif %}</option>
            {% endfor %}
        </select>

        <label>Vitesse d'élocution :</label>
        <div class="radio-group">
            <label class="radio-option">
                <input type="radio" name="speed" value="fast" checked>
                ⚡ Rapide (Style Série)
            </label>
            <label class="radio-option">
                <input type="radio" name="speed" value="slow">
                🐢 Lent (Style Jeu Original)
            </label>
        </div>

        <label for="text">Texte de la réplique :</label>
        <textarea id="text" placeholder="Entrez le dialogue ici..."></textarea>

        <button id="generateBtn" onclick="generateAudio()">Générer la Voix</button>
        <div id="loader" class="loader">Génération en cours... ⏳</div>

        <div id="resultSection" class="result-section">
            <label>Pré-écoute :</label>
            <audio id="audioPlayer" controls></audio>
            <a id="downloadLink" class="download-btn" href="#" download="replique.wav">📥 Télécharger le fichier .wav</a>
        </div>
    </div>

    <script>
        async function generateAudio() {
            const character = document.getElementById('character').value;
            const text = document.getElementById('text').value;
            const speed = document.querySelector('input[name="speed"]:checked').value;
            const btn = document.getElementById('generateBtn');
            const loader = document.getElementById('loader');
            const resultSection = document.getElementById('resultSection');
            const audioPlayer = document.getElementById('audioPlayer');
            const downloadLink = document.getElementById('downloadLink');

            if (!text.trim()) { alert("Veuillez entrer du texte !"); return; }

            btn.disabled = true;
            loader.style.display = 'block';
            resultSection.style.display = 'none';

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ character, text, speed })
                });

                if (!response.ok) throw new Error("Erreur lors de la génération");

                const filename = response.headers.get('x-filename') || 'replique.wav';
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                
                audioPlayer.src = url;
                downloadLink.href = url;
                downloadLink.download = filename;
                
                resultSection.style.display = 'flex';
                audioPlayer.play();
            } catch (error) {
                alert(error.message);
            } finally {
                btn.disabled = false;
                loader.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, characters=CHARACTERS)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    character_name = data.get('character', 'Tom Nook')
    text = data.get('text', '')
    speed_mode = data.get('speed', 'fast')
    
    # Récupération des données du personnage
    char_info = CHARACTERS.get(character_name, {"pitch": 1.0, "gender": "boy"})
    pitch_val = char_info["pitch"]
    gender = char_info["gender"]
    speed_factor = SPEED_FAST if speed_mode == 'fast' else SPEED_SLOW
    
    # Sélection de la bonne bibliothèque de phonèmes
    library_to_use = phoneme_libs.get(gender, phoneme_libs.get('boy', {}))
    
    # Génération
    final_signal = generate_animalese_v31(text, library_to_use, pitch_val, speed_factor, 44100)
    
    wav_io = io.BytesIO()
    wavfile.write(wav_io, 44100, final_signal)
    wav_io.seek(0)
    
    char_clean = character_name.lower().replace(" ", "_")
    text_clean = clean_filename_part(text) or "replique"
    filename = f"{char_clean}_{text_clean}.wav"
    
    response = send_file(wav_io, mimetype="audio/wav")
    response.headers["x-filename"] = filename
    response.headers["Access-Control-Expose-Headers"] = "x-filename"
    return response

if __name__ == "__main__":
    print("=======================================================")
    print("--- DÉMARRAGE DU SERVEUR DE PRODUCTION ANIMALESE ---")
    print("=======================================================\n")
    

    port = int(os.environ.get("PORT", 5000))
    

    app.run(host='0.0.0.0', port=port, debug=False)