# SummariV — Multimodal RAG for Video Question Answering

SummariV is a Multimodal Retrieval-Augmented Generation (RAG) framework for evidence-grounded question answering on educational videos. Given a natural language query, the system retrieves the most relevant audio and visual segments from a video, generates a grounded answer, and extracts the exact temporal clip as evidence.

---

## 🧠 How It Works

```
Video Input
    │
    ├─► Whisper ASR ──► Timestamped Transcript ──► SBERT Embeddings ──► FAISS (Text Index)
    │
    └─► OpenCV Frames ──► BLIP Captions ──────────► SBERT Embeddings ──► FAISS (Visual Index)
                                                              │
                                              Query Embedding ──► Top-k Retrieval
                                                              │
                                              Flan-T5 / LLM Answer Generation
                                                              │
                                              FFmpeg Temporal Clip Extraction
```

---

## 🏗️ Project Structure

```
SummariV/
│
├── SummariV_Unified.ipynb          # Full pipeline: Colab mode + FastAPI + ngrok
│
├── frontend/
│   └── index.html                  # Web UI (drag-drop upload, clip player)
│
├── evaluation/
│   ├── GroundTruth.csv             # 44 QA pairs (EDU-VSUM dataset)
│   ├── hyperparameter_tuning.ipynb # 54-config pipeline grid search
│   └── model_param_tuning.ipynb    # 96-config model parameter grid search
│
├── data/                           # Git-ignored — created at runtime by notebook
│   ├── video/
│   ├── frames/
│   ├── transcripts/
│   ├── embeddings/
│   └── clips/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Models Used

| Component | Model |
|---|---|
| Audio Transcription | OpenAI Whisper (base) |
| Visual Captioning | Salesforce BLIP (blip-image-captioning-base) |
| Sentence Embeddings | SBERT all-MiniLM-L6-v2 |
| Vector Search | FAISS (IndexFlatL2) |
| Answer Generation | Google Flan-T5-base (local fallback) |
| Clip Extraction | FFmpeg |

---

## 🚀 Getting Started

### Run on Google Colab (Recommended)

1. Open `SummariV.ipynb` in Google Colab.
2. Enable GPU runtime: `Runtime → Change runtime type → T4 GPU`.
3. The notebook has two independent parts:

**Part A — Colab Mode (Cells 1 → 16)**
Run directly inside the notebook. Upload a video when prompted, then enter your question. The answer and extracted clip are displayed inline.

**Part B — Web Backend (Cells 17 → 19)**
Exposes the same pipeline as a REST API via FastAPI + ngrok. After Cell 19 runs, it prints a public ngrok URL. Paste that URL into `frontend/index.html` and open the file in your browser to use the full web interface.

> Parts A and B share the same model loading (Cells 1–4). You do not need to reload models when switching between them.

### Local Setup

```bash
git clone https://github.com/<your-username>/SummariV.git
cd SummariV
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
```

> Local execution without a GPU will be significantly slower. Google Colab with a T4 GPU is strongly recommended.

---

## 📦 Requirements

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

FFmpeg must be installed separately:

```bash
# Linux / Colab
sudo apt-get install -y ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

---

## 🖥️ Web Interface

The frontend (`frontend/index.html`) provides:

- Drag-and-drop video upload
- Real-time pipeline stage tracker (Upload → Transcription → Captioning → Indexing → Ready)
- Natural language question input
- Answer card with confidence score
- Embedded video clip player showing the exact grounded evidence segment
- Full video summary generation

**To use the web interface:**
1. Run Cells 1–4 in `SummariV_Unified.ipynb` to load all models.
2. Run Cells 17–19 (Part B) to start the FastAPI server and get the ngrok URL.
3. Open `frontend/index.html` in your browser — the UI connects to the ngrok URL automatically.

---

## 🔑 API Keys

The notebook supports multiple LLM providers for answer generation and summarisation. Set your provider in **Cell 4**:

```python
API_PROVIDER = "local"   # Options: "anthropic" | "local"
```

| Provider | Key Variable | Where to Get |
|---|---|---|
| Anthropic Claude | `ANTHROPIC_API_KEY` | https://console.anthropic.com |
| Local (Flan-T5) | — | No key needed, runs on-device |

Paste your chosen API key directly into Cell 4 before running.

> **Never commit API keys to GitHub.** Add any config files containing keys to `.gitignore`.

---

## 📊 Evaluation Results

Evaluated on the **EDU-VSUM** dataset with 44 manually annotated question-answer pairs.

| Metric | Score |
|---|---|
| Answer Relevance (AR) | 0.6705 |
| Evidence Grounding Score (EGS) | 0.6705 |
| Temporal Localisation Accuracy (TLA) | 1.0000 |

### Best Hyperparameter Configuration

Identified via a 54-configuration pipeline grid search (see `evaluation/hyperparameter_tuning.ipynb`).

| Parameter | Optimal Value |
|---|---|
| Top-k Retrieval | 3 |
| Max Answer Tokens | 128 |
| Frame Extraction Interval | 1 second |
| Caption Max Tokens | 40 |

---

## 🔌 API Endpoints

When running in Part B (web backend) mode, the following REST endpoints are available:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status and active LLM provider |
| POST | `/upload` | Upload a video file to begin processing |
| GET | `/status` | Pipeline processing status and stage |
| POST | `/query` | Submit a question, returns answer + clip ID |
| GET | `/clip/{clip_id}` | Stream the extracted evidence clip |
| GET | `/summary` | Generate a full summary of the video |

---

## 📌 Technical Notes

- The system targets **specific factual queries** grounded to video timestamps, not open-ended summarisation.
- A `word_overlap` guard (threshold: 40%) prevents hallucinated answers from Flan-T5 — if overlap with retrieved context is too low, the system falls back to the raw retrieved segment directly.
- An L2 distance threshold (`L2_THRESHOLD = 1.2`) filters out low-confidence retrievals and returns a "not found in video" response instead of a hallucinated answer.
- Context length is capped per LLM provider to stay within token limits (Anthropic: ~4000 chars, local Flan-T5: ~500 chars).
- ROUGE metrics were found unsuitable for this system's evaluation profile; primary metrics are Answer Relevance (AR), Evidence Grounding Score (EGS), and Temporal Localisation Accuracy (TLA).

---

## 👥 Team

Developed as a final-year B.Tech (CSE) project at **PVPSIT, Vijayawada**.

---

## 📄 License

This project is for academic purposes. Please cite appropriately if you use this work.
