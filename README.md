# 🎙️ Animalese Studio V40 — Animal Crossing: New Leaf Voice Generator

Welcome to **Animalese Studio V40**, a Python-powered web application designed to generate authentic, dynamic *Animalese* voices (the iconic synthesized speech from the *Animal Crossing* series). 

This tool is specifically calibrated using the phoneme libraries extracted from **Animal Crossing: New Leaf** to create realistic dialogues for custom animations, video games, or fan series.

---

## 🚀 Live Demo

You can test the voice generator directly in your browser without installing anything:
👉 **[Try the Live Demo on Render](https://animalese-acnl.onrender.com/)**

> ⚠️ **Note:** The live version runs on a free cloud hosting tier. If the page takes a few moments to load initially, the server is just waking up!

---

## ✨ Features

* **Dynamic Pitch Engine:** Recreates natural speech inflections with macro-arcs and micro-bounces. It automatically adjusts intonation contextually based on punctuation (e.g., rising pitch for questions `?`, excited peaks for exclamations `!`).
* **Vowel Reduction Algorithm:** Automatically reduces consecutive vowels to mimic the signature high-speed chatter of the original Nintendo 3DS game.
* **Dual-Gender Phoneme Libraries:** Uses separated `BOY` and `GIRL` sound banks to stay true to each character's in-game database profile.
* **Speed Modulation:** * ⚡ **Fast Mode:** Optimized for animated series and modern video formats.
    * 🐢 **Slow Mode:** Recreates the classic, nostalgic pacing of the original video game.
* **Instant WAV Export:** Generates clean, normalized 16-bit 44.1kHz PCM WAV files ready to be downloaded and imported into any DAW or video editing software (like BandLab, Blender, or Unreal Engine).

---

## 🛠️ Tech Stack & Dependencies

The backend engine is lightweight and built entirely with Python:
* **Flask:** Web framework and API routing.
* **Librosa:** Advanced audio processing and RAM-loading.
* **NumPy & SciPy:** High-performance signal manipulation, mathematical modeling for pitch-shifting, and audio buffering.
* **Gunicorn:** Production-ready WSGI HTTP server for stable web deployment.

---

## 💻 Local Installation (For Developers)

If you want to run this studio locally on your machine or add custom characters:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Tscha74800/animalese-acnl.git](https://github.com/Tscha74800/animalese-acnl.git)
   cd animalese-acnl
