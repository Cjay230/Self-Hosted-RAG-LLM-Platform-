# Project context

## What this is

A self-hosted RAG system for corporate meetings, built for EECE 490 at AUB by
Carla Jaffal and Hadi Shmaisani. Meeting audio and scanned documents come in; the
system transcribes, generates minutes, extracts action items to Excel, translates
between English, Arabic and French, and answers questions over everything with
citations. It runs entirely offline on a laptop — no cloud API at inference time.

Alongside the system we measure it: accuracy against a fabricated test set, a
comparison against a cloud LLM, and a quantization cost table. That measurement
layer is the point of the project as much as the app is.

## Stack

- **Ollama** serving **Qwen2.5** 7B and 3B, 4-bit
- **Whisper** for speech to text, **EasyOCR** for scans
- **scikit-learn** TF-IDF, **sentence-transformers** `multilingual-e5-small`, **FAISS**
- **Streamlit** for the app, **pandas** and **openpyxl** for Excel output

## Pipeline

```
audio ──> Whisper ──┐
scans ──> OCR ──────┼──> chunks.jsonl ──> index ──> retrieve ──> grounded prompt ──> answer + citations
docs ───────────────┘                                    ^
                                                    access control (SVM)
```

## Conventions

**Chunk record** — one JSON object per line in `chunks.jsonl`:

```json
{"chunk_id": "m01-003", "doc_id": "meeting_01", "source_type": "transcript",
 "date": "2026-01-12", "language": "en", "speaker": "RANA",
 "text": "...", "confidential": false}
```

`source_type` is `transcript`, `document`, `scan` or `minutes`.
`language` is `en`, `ar` or `fr`.

**Function signatures** are fixed in `INTERFACES.md`. Read it before writing any
module. Never change a signature — if one looks wrong, say so and stop.

**Results** go to `results/<component>.csv` with columns
`component, setting, metric, value, date`. One file per component.
Never a single shared results file.

## File ownership

Two people work in this repo and must not edit each other's files.

| Path | Owner |
|---|---|
| `src/capture.py`, `src/build_index.py`, `src/retrieval.py`, `src/security.py`, `eval/` | Carla |
| `src/llm.py`, `src/router.py`, `src/minutes.py`, `src/translate.py`, `app.py`, `bench/` | Hadi |
| `data/`, `facts.csv` | Carla |
| `INTERFACES.md`, `README.md`, `CLAUDE.md` | Both |

Before creating or editing a file, check this table. If a task would require
touching a file owned by the other person, say so and stop rather than doing it.

## How to work in this repo

1. **One task at a time.** Do what is asked and nothing beyond it. Do not scaffold
   future components, do not implement functions that were not requested, do not
   "while I'm here" anything.

2. **Simple over clever.** The two authors must be able to explain every line of
   this code in an oral exam. Prefer a readable loop to a dense comprehension.
   Avoid abstractions that exist only for elegance.

3. **Comment the why, not the what.** When a choice has an alternative — cosine
   similarity over dot product, 500-token chunks over 200 — say in a comment why
   this one.

4. **No new dependencies** without saying so first. The install list is fixed and
   the network here is slow.

5. **Ask before assuming.** If a requirement is ambiguous, ask rather than
   picking an interpretation and building on it.

## Course constraints

- Must run on Colab as well as locally; notebooks must execute top to bottom in a
  fresh session.
- The LLM is not trained or fine-tuned. It is used as-is. The only trained models
  are small: the query router, the confidentiality SVM, and a page-type CNN.
- Every claim in the report needs a number behind it from `results/`.
