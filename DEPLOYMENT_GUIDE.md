# Singing Module Deployment Guide (Beginner Friendly)

This document explains **everything** you need to deploy your singing module as a live demo link for your resume.

You do **not** need prior Docker experience. Follow steps in order.

---

## 1) What we are trying to achieve

**Goal:** Put a public URL on your resume like:

`https://huggingface.co/spaces/YOUR_USERNAME/singing-evaluation-demo`

When a recruiter opens it, they can:
- upload reference + user audio
- click Analyze
- see scores, pitch feedback, and live note tracking

This proves your module works without waiting for the full integrated project.

---

## 2) Big-picture architecture

```mermaid
flowchart LR
    A[Your code on laptop] --> B[GitHub repo]
    B --> C[Hugging Face Space]
    C --> D[Docker container runs app]
    D --> E[Public demo URL for resume]
```

What each part does:

| Piece | What it is | Why we use it |
|---|---|---|
| **Streamlit app (`app.py`)** | Web UI | Easy demo interface |
| **Analyzer (`analyzer.py`)** | ML backend | Pitch/rhythm/lyrics logic |
| **GitHub** | Code hosting | Version control + source for deployment |
| **Docker** | Packaging tool | Bundles app + Python + ffmpeg + libraries |
| **Hugging Face Spaces** | Free ML hosting | Reliable public demo links for AI projects |

---

## 3) What is Docker? (simple explanation)

Think of Docker like a **lunchbox**:
- Your app is the food.
- Python + libraries + ffmpeg are utensils/ingredients.
- Docker packs all of it together so it runs the same on any machine/cloud.

You don't manually install everything on Hugging Face servers.  
You give them the lunchbox (`Dockerfile`), and they run it.

### Files related to Docker in this folder

| File | Purpose |
|---|---|
| `Dockerfile` | Recipe to build the runtime environment |
| `requirements.txt` | Python packages to install |
| `.dockerignore` | Files to exclude from container build |
| `.streamlit/config.toml` | Streamlit server settings for cloud |

---

## 4) Technologies used in this module

- **Python 3.10**
- **Streamlit** — frontend UI
- **PyTorch (CPU)** — deep learning runtime
- **torchcrepe (CREPE)** — pitch detection
- **Demucs** — vocal separation for reference audio
- **OpenAI Whisper** — lyrics transcription
- **Librosa** — audio loading, onset/rhythm, DTW alignment
- **Matplotlib** — pitch contour plot
- **ffmpeg** — mp3/audio decoding support in Linux container

---

## 5) Prerequisites (install once)

You need 3 accounts/tools:

1. **GitHub account** — https://github.com/signup  
2. **Hugging Face account** — https://huggingface.co/join  
3. **Git** installed on Windows — https://git-scm.com/download/win

Optional but helpful:
- **GitHub Desktop** (if command line feels hard)

---

## 6) Files already prepared in this folder

These were created/updated for deployment:

- `Dockerfile`
- `requirements.txt` (added missing `torchcrepe`, removed duplicate torch install)
- `README.md` (Hugging Face Space metadata + project intro)
- `.streamlit/config.toml`
- `.gitignore`
- `.dockerignore`
- `DEPLOYMENT_GUIDE.md` (this file)

---

## 7) Step-by-step deployment (recommended path)

## Step A — Create GitHub repository

1. Go to GitHub → **New repository**
2. Name: `singing-evaluation-demo`
3. Visibility: Public
4. Do **not** initialize with README (we already have one locally)
5. Create repo

---

## Step B — Push local code to GitHub

Open PowerShell in this folder:

`D:\Downloads_D\SINGING_MODULE`

Run:

```powershell
git init
git add .
git commit -m "Initial singing module demo for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/singing-evaluation-demo.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME`.

If Git asks for login, use GitHub username + Personal Access Token (not password).

---

## Step C — Create Hugging Face Space

1. Open https://huggingface.co/new-space
2. Space name: `singing-evaluation-demo`
3. License: MIT (or your choice)
4. Select SDK: **Docker**
5. Hardware: **CPU basic** (free)
6. Create Space

---

## Step D — Connect Space to GitHub (easiest)

In your new Space page:

1. Go to **Settings**
2. Find **Repository** / **Sync with GitHub** (wording may vary)
3. Connect your GitHub repo: `singing-evaluation-demo`
4. Save

Hugging Face will automatically build your Docker image.

---

## Step E — Alternative: push directly to Hugging Face (if GitHub sync unavailable)

1. Create HF Access Token:  
   https://huggingface.co/settings/tokens  
   (Read + Write)

2. Clone your Space repo:

```powershell
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/singing-evaluation-demo
cd singing-evaluation-demo
```

3. Copy project files from `D:\Downloads_D\SINGING_MODULE` into this cloned folder  
   (except `.git` from old folder)

4. Commit + push:

```powershell
git add .
git commit -m "Deploy singing demo"
git push
```

When prompted:
- Username: your HF username
- Password: HF access token

---

## Step F — Wait for build

On Space page, open **Logs** tab.

Expected stages:
1. Building Docker image (10–20 min first time)
2. Installing Python packages
3. Starting Streamlit

When status becomes **Running**, open the App tab.

---

## Step G — Test like a recruiter

Use your known test files (`ref2.mp3`, `user2.mp3`):

1. Upload both
2. Click **Analyze**
3. Wait patiently (first run can take 2–4 minutes on free CPU)
4. Confirm:
   - scores appear
   - pitch plot appears
   - live note tracking works

If this works once, your resume link is viable.

---

## 8) What to put in resume/application

**Project line example:**

> Singing Evaluation Module (Live Demo) — Built pitch/rhythm/lyrics analysis pipeline with CREPE, Demucs, Whisper, and Streamlit UI; deployed on Hugging Face Spaces.

**Link format:**

`https://huggingface.co/spaces/YOUR_HF_USERNAME/singing-evaluation-demo`

**Interview explanation (30 sec):**

> I built the singing evaluation module of our larger music-learning pipeline. It compares user vocals against reference audio using pitch extraction, vocal separation, lyrics scoring, and timeline feedback. I deployed this module independently as a live demo while full-system integration is in progress.

---

## 9) Reliability tips (important for company evaluation)

Because this is heavy ML on free CPU:

1. Add this note in application form:
   - "First analysis may take 2–4 minutes due to model loading on CPU."
2. Keep a backup 45–60 sec screen recording (Loom/phone) showing successful run.
3. Test the link once right before submitting.
4. Avoid uploading very long audio (>2 min) during evaluation.

---

## 10) Troubleshooting

### Build fails at `pip install`
- Check Logs for missing package
- Ensure `requirements.txt` includes `torchcrepe`
- Re-run build from Space settings

### App opens but Analyze fails
- Try shorter mp3/wav files first
- Check Logs while clicking Analyze
- Whisper/Demucs may fail if memory is tight; app still should show useful pitch results in many cases

### "Repository not found" on git push
- Verify remote URL
- Ensure token permissions are correct

### Space keeps restarting
- Usually memory pressure from large models
- Retry with shorter audio
- Consider upgrading HF hardware temporarily (if available)

---

## 11) Command cheat sheet

```powershell
# from project folder
git status
git add .
git commit -m "Update demo"
git push

# clone HF space
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/singing-evaluation-demo

# run locally (optional test before deploy)
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
streamlit run app.py
```

---

## 12) Tonight checklist (for 10 AM deadline)

- [ ] GitHub repo created
- [ ] Code pushed
- [ ] HF Space created (Docker)
- [ ] Build completed (Running)
- [ ] Test analyze once successfully
- [ ] Resume link added
- [ ] Backup demo video uploaded (Google Drive/Loom)

---

## 13) Need help fast?

If build fails, send:
1. HF Space link
2. Screenshot/text of Logs error
3. Whether GitHub push succeeded

Then we can patch Dockerfile/requirements quickly.
