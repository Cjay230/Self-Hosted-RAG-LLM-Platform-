# INTERFACES

Agreed between Carla and Hadi. **Nobody changes a signature without saying so in the
group chat first.** A silent change breaks the other person's work.

---

## The chunk record

Every piece of text in the system — transcript, document, scan, generated minutes —
becomes one JSON object per line in `chunks.jsonl`:

```json
{
  "chunk_id": "m01-003",
  "doc_id": "meeting_01",
  "source_type": "transcript",
  "date": "2026-01-12",
  "language": "en",
  "speaker": "RANA",
  "text": "We're deferring the warehouse expansion to Q3.",
  "confidential": false
}
```

- `source_type` is one of: `transcript`, `document`, `scan`, `minutes`
- `language` is `en`, `ar` or `fr`
- `speaker` is null for documents and scans
- `confidential` is set by `is_confidential()`, not by hand

---

## Carla's functions

```python
# src/capture.py
def transcribe(audio_path: str, translate_to: str = "en") -> dict:
    """Whisper transcription. Returns
    {"text": str, "language": str, "segments": list}"""

def parse_transcript(path: str) -> tuple[dict, list[dict]]:
    """Read a meeting file. Returns (doc_meta, turns) where
    doc_meta = {"doc_id", "date", "language", "source_type", "attendees"}
    turns    = [{"speaker": "RANA", "text": "..."}, ...]"""

def chunk_turns(turns: list[dict], doc_meta: dict,
                target_words: int = 150, overlap_words: int = 30) -> list[dict]:
    """Chunk a transcript into chunk records.
    Never splits a speaker's turn across two chunks — if a turn would
    overflow the target, start a new chunk instead. A planted fact that
    gets cut in half can never be retrieved intact.
    speaker is set when the chunk holds one turn, null when it spans several."""

def chunk_document(text: str, doc_meta: dict,
                   target_words: int = 150, overlap_words: int = 30) -> list[dict]:
    """Chunk a document or OCR page. Splits on paragraph, then sentence.
    Never splits mid-sentence. speaker is always null."""
```

Word counts for now. The real Qwen tokenizer replaces them once Ollama is
installed — Arabic costs more tokens per word than English, which matters for
the context budget.

```python
# src/retrieval.py
def retrieve(query: str, k: int = 5, method: str = "hybrid") -> list[dict]:
    """method is "tfidf", "dense" or "hybrid".
    Returns k chunk records, each with an added "score" float, best first."""
```

```python
# src/security.py
def is_confidential(chunk_text: str) -> bool:
    """Linear SVM. True means hide from users without clearance."""

def is_anomalous_query(query: str) -> tuple[bool, float]:
    """Gaussian detector. Returns (flagged, score).
    True means reject the query before it reaches the LLM."""
```

---

## Hadi's functions

```python
# src/llm.py
def ask(prompt: str, model: str = "qwen2.5:7b") -> str:
    """Raw call to Ollama. No retrieval, no grounding."""

def answer(query: str, chunks: list[dict], model: str = "qwen2.5:7b") -> dict:
    """Grounded answer. Returns
    {"text": str, "citations": list[str], "latency_s": float}
    citations are chunk_ids. Must return "I don't know" when the
    answer is not in the chunks."""
```

```python
# src/capture.py  (Hadi owns this one function only)
def ocr_page(image_path: str) -> str:
    """EasyOCR. Returns the page text."""
```

```python
# src/minutes.py
def generate_minutes(transcript: str, language: str = "en") -> str:
    """Markdown: attendees, agenda, decisions, action items, open questions."""

def extract_actions(transcript: str) -> list[dict]:
    """Returns [{"task", "owner", "due_date", "meeting", "status"}, ...]
    Ask the model for JSON and validate it — free text will not parse."""
```

```python
# src/translate.py
def translate(text: str, target_lang: str) -> str:
    """target_lang is "en", "ar" or "fr"."""
```

```python
# src/router.py
def route(query: str) -> str:
    """Logistic regression. Returns one of:
    "document_question", "translation", "summarization", "general".
    Used to send easy queries to the 3B model."""
```

---

## Results files

One file per component, one owner each. **Never a shared `results.csv`** — two people
appending to the same file conflicts every time.

| File | Owner |
|---|---|
| `results/capture.csv` | Carla |
| `results/retrieval.csv` | Carla |
| `results/evaluation.csv` | Carla |
| `results/security.csv` | Carla |
| `results/quantization.csv` | Hadi |
| `results/router.csv` | Hadi |
| `results/translation.csv` | Hadi |

Columns, the same in every file: `component, setting, metric, value, date`
