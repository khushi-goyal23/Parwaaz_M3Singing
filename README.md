---
title: Singing Evaluation Demo
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Singing Evaluation System (Demo)

Compare your singing against a reference track and get pitch, rhythm, lyrics, and live note feedback.

## How to use

1. Upload **Reference Audio** (original song/vocal).
2. Upload **Your Singing**.
3. Click **Analyze**.
4. First run may take **2-4 minutes** on cloud CPU (model download + vocal separation + transcription).

## Tech stack

- **Streamlit** — web UI
- **CREPE (torchcrepe)** — pitch extraction
- **Demucs** — vocal separation (reference track)
- **Whisper** — lyrics transcription
- **Librosa** — audio processing + alignment

## Resume note

This is the standalone singing-analysis module from a larger deep-learning music pipeline.
